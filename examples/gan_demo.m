% GAN demonstration for NNET_matlab
clear all
close all
clc

%% Create data set 
load ../data/mnist_all.mat
data = [train0;...
        train1;...
        train2;...
        train3;...
        train4;...
        train5;...
        train6;...
        train7;...
        train8;...
        train9];
labels_id =  [0*ones(size(train0,1),1);...
           1*ones(size(train1,1),1);...
           2*ones(size(train2,1),1);...
           3*ones(size(train3,1),1);...
           4*ones(size(train4,1),1);...
           5*ones(size(train5,1),1);...
           6*ones(size(train6,1),1);...
           7*ones(size(train7,1),1);...
           8*ones(size(train8,1),1);...
           9*ones(size(train9,1),1)];
labels = zeros(size(labels_id,1), 10);
for iLb = 0:9
    labels(labels_id == iLb, iLb+1) = 1;
end
% little visualization
X.data = data';
X.targets = labels'; 
%% Create networks
addpath('../../NNET_matlab');
run ../load_nnet_basic.m;


%%%% Discriminator network
dis_in_dim = size(X.data, 1);
hid_dim = 64;
out_dim = size(X.targets, 1);

dis_net.layers(1) = layer;
dis_net.layers(1).W = randn(hid_dim, dis_in_dim)/sqrt(dis_in_dim);
dis_net.layers(1).b = zeros(hid_dim, 1);
dis_net.layers(1).f = relu;
dis_net.layers(1).f_prime = relu_prime;
dis_net.layers(1).delta_W = zeros(hid_dim, dis_in_dim);
dis_net.layers(1).delta_b = zeros(hid_dim, 1);
dis_net.layers(1).grad_W = zeros(hid_dim, dis_in_dim);
dis_net.layers(1).grad_b = zeros(hid_dim, 1);

dis_net.layers(2) = layer;
dis_net.layers(2).W = randn(hid_dim, hid_dim)/sqrt(hid_dim);
dis_net.layers(2).b = zeros(hid_dim, 1);
dis_net.layers(2).f = relu;
dis_net.layers(2).f_prime = relu_prime;
dis_net.layers(2).delta_W = zeros(hid_dim, hid_dim);
dis_net.layers(2).delta_b = zeros(hid_dim, 1);
dis_net.layers(2).grad_W = zeros(hid_dim, hid_dim);
dis_net.layers(2).grad_b = zeros(hid_dim, 1);


dis_net.layers(3) = layer;
dis_net.layers(3).W = randn(out_dim, hid_dim)/sqrt(hid_dim);
dis_net.layers(3).b = zeros(out_dim, 1);
dis_net.layers(3).f = logistic;
dis_net.layers(3).f_prime = logistic_prime;
dis_net.layers(3).delta_W = zeros(out_dim, hid_dim);
dis_net.layers(3).delta_b = zeros(out_dim, 1);
dis_net.layers(3).grad_W = zeros(out_dim, hid_dim);
dis_net.layers(3).grad_b = zeros(out_dim, 1);

%%%% Generator network
gen_in_dim = 20;
hid_dim = 64;
out_dim = size(X.data, 1);

gen_net.layers(1) = layer;
gen_net.layers(1).W = randn(hid_dim, gen_in_dim)/sqrt(gen_in_dim);
gen_net.layers(1).b = zeros(hid_dim, 1);
gen_net.layers(1).f = relu;
gen_net.layers(1).f_prime = relu_prime;
gen_net.layers(1).delta_W = zeros(hid_dim, gen_in_dim);
gen_net.layers(1).delta_b = zeros(hid_dim, 1);
gen_net.layers(1).grad_W = zeros(hid_dim, gen_in_dim);
gen_net.layers(1).grad_b = zeros(hid_dim, 1);

gen_net.layers(2) = layer;
gen_net.layers(2).W = randn(hid_dim, hid_dim)/sqrt(hid_dim);
gen_net.layers(2).b = zeros(hid_dim, 1);
gen_net.layers(2).f = relu;
gen_net.layers(2).f_prime = relu_prime;
gen_net.layers(2).delta_W = zeros(hid_dim, hid_dim);
gen_net.layers(2).delta_b = zeros(hid_dim, 1);
gen_net.layers(2).grad_W = zeros(hid_dim, hid_dim);
gen_net.layers(2).grad_b = zeros(hid_dim, 1);


gen_net.layers(3) = layer;
gen_net.layers(3).W = randn(out_dim, hid_dim)/sqrt(hid_dim);
gen_net.layers(3).b = zeros(out_dim, 1);
gen_net.layers(3).f = logistic;
gen_net.layers(3).f_prime = logistic_prime;
gen_net.layers(3).delta_W = zeros(out_dim, hid_dim);
gen_net.layers(3).delta_b = zeros(out_dim, 1);
gen_net.layers(3).grad_W = zeros(out_dim, hid_dim);
gen_net.layers(3).grad_b = zeros(out_dim, 1);


%% Train the network
%% And Now the algorithm finally
batchsize = 100;
n_epochs = 200;
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