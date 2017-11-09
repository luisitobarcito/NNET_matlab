% XOR demonstration for NNET_matlab
clear all
close all
clc

%% Create data set 
N = 1000;
data = mvnrnd([0, 0], eye(2), N);
bin_idx = rand(N, 2) > 0.5;

data = data + bin_idx*4;
labels =  xor(bin_idx(:,1), bin_idx(:,2));
% little visualization
gscatter(data(:,1), data(:,2), labels)
X.data = data';
X.targets = labels'; 
%% Create network
addpath('../../NNET_matlab');
run ../load_nnet_basic.m;

in_dim = size(X.data, 1);
hid_dim = 6;
out_dim = size(X.targets, 1);

net.layers(1) = layer;
net.layers(1).W = randn(hid_dim, in_dim)/sqrt(in_dim);
net.layers(1).b = zeros(hid_dim, 1);
net.layers(1).f = hyptan;
net.layers(1).f_prime = hyptan_prime;
net.layers(1).delta_W = zeros(hid_dim, in_dim);
net.layers(1).delta_b = zeros(hid_dim, 1);
net.layers(1).grad_W = zeros(hid_dim, in_dim);
net.layers(1).grad_b = zeros(hid_dim, 1);

net.layers(2) = layer;
net.layers(2).W = randn(out_dim, hid_dim)/sqrt(hid_dim);
net.layers(2).b = zeros(out_dim, 1);
net.layers(2).f = logistic;
net.layers(2).f_prime = logistic_prime;
net.layers(2).delta_W = zeros(out_dim, hid_dim);
net.layers(2).delta_b = zeros(out_dim, 1);
net.layers(2).grad_W = zeros(out_dim, hid_dim);
net.layers(2).grad_b = zeros(out_dim, 1);

%% Train the network
%% And Now the algorithm finally
batchsize = 1000;
n_epochs = 600;
stepsize = 0.1;
[X_batches] = createMiniBatches(X, batchsize);
n_batches = size(X_batches.data, 3);
total_cost = zeros(n_batches, n_epochs);
figure(2)
for iEpc = 1:n_epochs
    fprintf('Epoch  %d\n',iEpc);
    for iBtch = 1 : n_batches
        
        net.layers = propagateForward(net.layers, X_batches.data(:,:, iBtch));
        % sample from code distribution
        Pred_label = net.layers(end).X_out;
        Pred_err = Pred_label - X_batches.targets(:,:,iBtch);
        total_cost(iBtch, iEpc) = (1/batchsize)*sum(-X_batches.targets(:,:,iBtch).*log(Pred_label) - (1-X_batches.targets(:,:,iBtch)).*log(1- Pred_label));
        diff_cost = (1/batchsize)*Pred_err./(Pred_label.*(1-Pred_label));
        % backpropagate the gradients
        net.layers = propagateBackward(net.layers, diff_cost);
        % compute updates for decoder networks
        net.layers = updateParamters(net.layers, stepsize, 'sgd');
        if mod(iBtch, 1)  == 0
            net.layers = propagateForward(net.layers, X.data);
            Pred_temp = net.layers(end).X_out;
            gscatter(X.data(1,:), X.data(2,:), Pred_temp > 0.5 )
            xlim([-4 8])
            ylim([-4 8])
            drawnow;
            pause(0.01)
        end
    end
    fprintf('Cost is %f\n', mean(total_cost(:, iEpc)));
end