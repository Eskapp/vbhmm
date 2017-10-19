function [net,astar,bstar,pistar,u,v,wa,wb,wpi,Alph_init]= vbhmm_contDir(data,M,K,Alph_init,its,tol,net)

%Variational Bayesian Dirichlet-based Hidden Markov Model

%% DEBUG -------------------------
% M = 2;
% K = 3;
% N = 4;
% A = zeros(M,N,K);
% A(:,:,1) = [12 30 45 2;32 50 16 50];
% A(:,:,2) = [55 28 35 10;3 118 60 4];
% A(:,:,3) = [25 18 90 10;75 2 80 20];
% B = [.3 .3 .4;.4 .2 .4;.4 .3 .3];
% C = [.5 .5;.5 .5;.5 .5];
% Pi = [.8 .1 .1]';
% Number = 2000; 
% T = ones(1,Number)*20;
% [data,RecordingStateMix]= GenerateHMMDsamples(A,B,C,Pi,T,Number);
%% END DEBUG ---------------------

if nargin<7,
    initfromprior = 1;
else
    initfromprior = 0;
end;
if nargin<6,
    tol = 0.001;
end;
if nargin<5,
    its = 100; % Initially set to 100
end;

N = size(data,2); % Number of SEQUENCES to train on
T = zeros(1,N); % Number of observations per sequence
D = size(data{1},1); % data dimension
for n = 1:N,
    T(n) = size(data{n},2);
end;
total_length = sum(T);

% Initialise the hyperparameters
alphaa=1; % Hyperparameter for the transition matrix
alphab=1; % Hyperparameter for the mixing matrix
alphapi=1; % Hyperparameter for initial state pmf
% Initialise the pseudo-counts
ua = ones(1,K)*(alphaa/K); % Hyperparameter for the transition matrix
ub = ones(1,M)*(alphab/M); % Hyperparameter for the mixing matrix
upi = ones(1,K)*(alphapi/K); % Hyperparameter for initial state pmf
% Pick an HMM from the prior to initialize the counts
wa = zeros(K); wb = zeros(K,M);
for k=1:K % loop over hidden states
    wa(k,:) = dirrnd(ua,1)*total_length; % Transition matrix
    wb(k,:) = dirrnd(ub,1)*total_length; % Mixing matrix
end
wpi = dirrnd(upi,1)*total_length; % Initial state pmf
% Alldata = [];
% for n = 1:N
%    Alldata = cat(2,Alldata,data{n});
% end
% Alph_init = MomentMatchingDM(K*M,Alldata);
% while sum(isinf(Alph_init(:)))~=0
%     Alph_init = MomentMatchingDM(K*M,Alldata);
% end

sumAlphinit = zeros(K,N); % This parameter allows us to keep a raisonnable value for the variance of the estimated alpha.
% Alph_init = permute(reshape(Alph_init',[D M K]),[2 1 3]);
for k=1:K
    for m=1:M
        sumAlphinit(k,m) = sum(Alph_init(m,:,k)); 
    end
end

sumAlphinitRep = zeros(K,M,D);
for d=1:D
    sumAlphinitRep(:,:,d) = sumAlphinit;
end

%% Initialization of the hyperparameters
% u = 1 + zeros(size(permute(Alph,[3 1 2])));
% v = 0.01 + zeros(size(u));
u = Alph_init; % Try #1 - u./v = AlphaInit
v = 1 + zeros(size(u)); % Try #1
sumUV = sumAlphinitRep;

%% Log Data computation
LogData=cell(1,N);
for n=1:N
    LogData{n} = log(data{n});
end

Fold = -Inf; ntol = tol*N;

for it=1:its
%     tStart = tic;
    if (initfromprior==0 && it==1)
        Wa=hmm.Wa; % Transition matrix
        Wb=hmm.Wb; % Mixing matrix
        Wpi=hmm.Wpi; % Initial state pmf
        disp('initing');
    else
        % M Step
        Wa = wa + repmat(ua,[K 1]); % posterior is data counts plus prior.
        Wb = wb + repmat(ub,[K 1]);
        Wpi = wpi + upi;
    end
    
    astar = exp(digamma(Wa) - repmat(digamma(sum(Wa,2)),[1 K]));
    bstar = exp(digamma(Wb) - repmat(digamma(sum(Wb,2)),[1 M]));
    pistar = exp(digamma(Wpi) - digamma(sum(Wpi)));
    
    % E Step
    Gm = {}; Xi = {}; FWDvar = {};
    Zv = zeros(1,N);
    cumPi = zeros(K,1);
    cumXi = zeros(K,K);
    cumGm = zeros(K,M);
    
    %% Computation of the weighted log data vector
    NX = zeros(D,K,M); % Rappel : D est ici la dimension des data
    for n=1:N
        obslik = dataLikelihood_DM(((u./v).*permute(sumAlphinitRep,[2 3 1]))./permute(sumUV,[2 3 1]),data{n}); % u./v doit etre divise par sa propre somme aussi
        [Gm{n},Xi{n},FWDvar{n}] = forback(astar,bstar,pistar,obslik); %#ok<AGROW>
        Zv(n) = prod(FWDvar{n});
        for m=1:M
            for k=1:K
                NX(:,k,m) = NX(:,k,m) + LogData{n}*Gm{n}(:,m,k);
            end
        end
        %% Calculer cumXi, cumGm, cumPi
        cumPi = cumPi + permute(sum(Gm{n}(1,:,:),2),[3 2 1]);
        cumXi = cumXi + permute(sum(Xi{n},1),[2 3 1]);
        cumGm = cumGm + permute(sum(Gm{n},1),[3,2,1]);
    end
    lnZ(it) = log(sum(Zv)); %#ok<AGROW>
    
    %% F(alpha)
    phi = zeros(M,D,K);
    for k = 1:K
        for m = 1:M
            for d = 1:D
                alphaBar = u./v; % Manque une division par la somme des u./v
                phi(m,d,k) = cumGm(k,m)/(sum(T))*(alphaBar(k,m,d))*...
                    ( psi(sum(alphaBar(k,m,:))) - psi(alphaBar(k,m,d)) + sum( psi(1,sum(alphaBar(k,m,:))) * alphaBar(k,m,:).*( psi(u(k,m,:)) - log(v(k,m,:)) - log(alphaBar(k,m,:)) ) )...
                    - ( psi(1,sum(alphaBar(k,m,d))) * alphaBar(k,m,d)*( psi(u(k,m,d)) - log(v(k,m,d)) - log(alphaBar(k,m,d)) ) ) );
            end
        end
    end
    u = u+phi;
    v = v - permute(NX./repmat(cumGm,[1 1 D]),[3 1 2]); % v is a [M D K] array, NX is a [D K M] array
    
    % wa, wb, wpi peuvent etre calcules comme suit:
    wpi = cumPi'/sum(cumPi);
    wa = inv(diag(sum(cumXi,2))) * cumXi; %#ok<MINV>
    wb = inv(diag(sum(cumGm,2))) * cumGm; %#ok<MINV>
 
    Fa(it)=0; Fb(it)=0; Fpi(it)=0; %#ok<AGROW>
    for kk = 1:K,
        Fa(it) = Fa(it) - kldirichlet(Wa(kk,:),ua); %#ok<AGROW> %  les termes qui dependent des pdfs sont dans lnZ.
        Fb(it) = Fb(it) - kldirichlet(Wb(kk,:),ub); %#ok<AGROW>
    end
    Fpi(it) = - kldirichlet(Wpi,upi); %#ok<AGROW>
    
    F(it) = Fa(it)+Fb(it)+Fpi(it)+lnZ(it); %#ok<AGROW>
    
%     ElapsedTime = toc(tStart);
    
    if it == 1
        fprintf('It:%3i \tFa:%3.3f \tFb:%3.3f \tFpi:%3.3f \tFy:%3.3f \tF:%3.3f\n',it,Fa(it),Fb(it),Fpi(it),lnZ(it),F(it));
    else
        fprintf('It:%3i \tFa:%3.3f \tFb:%3.3f \tFpi:%3.3f \tFy:%3.3f \tF:%3.3f \tdF:%3.3f\n',it,Fa(it),Fb(it),Fpi(it),lnZ(it),F(it),F(it)-Fold);
    end

    Fold = F(it);
    if (it>2)
%         if abs(F(it)-F(it-1))<1e-6;
        if (F(it)<(F(it-1) - 1e-6)),     fprintf('violation');
        elseif ((F(it)-F(2))<(1 + ntol)*(F(it-1)-F(2))||~isfinite(F(it)))
            fprintf('\nconverged\nend\n');    break;
        end
    end
end

net.Wa = Wa;
net.Wb = Wb;
net.Wpi = Wpi;
net.F = [(1:size(F,2))' Fa' Fb' Fpi' lnZ' F'];
