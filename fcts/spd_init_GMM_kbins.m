function model = spd_init_GMM_kbins(Data, model, nbSamples, spdDataId)
% Noémie Jaquier, 2018
%
% This function computes K-Bins initialisation for GMM on the SPD manifold.
%
% Parameters:
%   - Data:         Set of data in vector form. Some dimensions of these
%                   data are SPD data expressed in Mandel notation. The
%                   remaining part of the data is Euclidean.
%   - model:        Model variable
%   - nbSamples:    Number of distinct demonstrations
%   - spdDataId:    Indices of SPD data in Data
% 
% Returns:
%   - model:        Initialized model

nbData = size(Data,2) / nbSamples;
if ~isfield(model,'params_diagRegFact')
	model.params_diagRegFact = 1E-4; %Optional regularization term to avoid numerical instability
end

% Delimit the cluster bins for the first demonstration
tSep = round(linspace(0, nbData, model.nbStates+1));

% Compute statistics for each bin
for i=1:model.nbStates
	id=[];
	for n=1:nbSamples
		id = [id (n-1)*nbData+[tSep(i)+1:tSep(i+1)]];
	end
	model.Priors(i) = length(id);
	
	% Mean computed on SPD manifold for parts of the data belonging to the
	% manifold
	if nargin < 4
		model.MuMan(:,i) = symmat2vec(spdMean(vec2symmat(Data(:,id))));
	else
		model.MuMan(:,i) = mean(Data(:,id),2);
		if iscell(spdDataId)
			for c = 1:length(spdDataId)
				model.MuMan(spdDataId{c},i) = ...
                    symmat2vec(spdMean(vec2symmat(Data(spdDataId{c},id)),3));
			end
		else
			model.MuMan(spdDataId,i) = ...
                symmat2vec(spdMean(vec2symmat(Data(spdDataId,id)),3));
		end
	end
	
	% Parts of data belonging to SPD manifold projected to tangent space at
	% the mean to compute the covariance tensor in the tangent space
	DataTgt = zeros(size(Data(:,id)));
	if nargin < 4
		DataTgt = logmap_vec(Data(:,id),model.MuMan(:,i));
	else
		DataTgt = Data(:,id);
		if iscell(spdDataId)
			for c = 1:length(spdDataId)
				DataTgt(spdDataId{c},:) = ...
                    logmap_vec(Data(spdDataId{c},id),model.MuMan(spdDataId{c},i));
			end
		else
			DataTgt(spdDataId,:) = ...
                logmap_vec(Data(spdDataId,id),model.MuMan(spdDataId,i));
		end
	end

	model.Sigma(:,:,i) = cov(DataTgt') + ...
        eye(model.nbVarVec).*model.params_diagRegFact;
	
end
model.Priors = model.Priors / sum(model.Priors);
end
