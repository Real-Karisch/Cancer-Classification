function bpmaintumor(input)
clc
%% Data Generation and Inputs

%%%%%%%% SET FOLDCOUNT %%%%%
FoldCount = 1;             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

half_samp = 1;

samp_size = 21000; %MAKE DIVISIBLE BY 3 FOR CROSS VAL

[in_r,in_c] = size(input);
one_ind = input(input(:, end) == 1,:);
zero_ind = input(input(:,end) == 0,:);

tumor_full = input;
%tumor_full = csvread('tumor_bin.csv', 1, 0);
if half_samp == 1
    tumor_samp_full = zeros(samp_size, in_c);
    tumor_samp_full(1:round(samp_size / 2),:) = datasample(one_ind,round(samp_size / 2));
    tumor_samp_full(round(samp_size / 2) + 1: end, :) = datasample(zero_ind, floor(samp_size / 2));
else
    tumor_samp_full = datasample(tumor_full, samp_size);
end
tumor_res_temp = zeros(samp_size, 1);
necrotic = find(tumor_samp_full(:,5) == 1);
recurrent = find(tumor_samp_full(:,5) == 0);

tumor_res_temp(necrotic,:) = 0;
tumor_res_temp(recurrent,:) = 1;

tumor_samp_full = [tumor_samp_full tumor_res_temp];

tumor_samp_train = tumor_samp_full(1:round(samp_size*2/3),:);
tumor_samp_test = tumor_samp_full(round(samp_size*2/3)+1:end,:);


%[train_actual, test_actual, complete_actual] = open_data_exam_1_NML_502;
%train = train_actual;
%test = test_actual;
%complete = complete_actual;

train = tumor_samp_train;
test = tumor_samp_test;
complete = tumor_samp_full;

[tr_r,tr_c] = size(train);
[tst_r, tst_c] = size(test);
[com_r, com_c] = size(complete);

train_res = train(:,5:6);
test_res = test(:,5:6);
complete_res = complete(:,5:6);

train(:,1:4) = train(:,1:4) / max(max(train(:, 1:4) * 0.95));
test(:,1:4) = test(:,1:4) / max(max(test(:, 1:4) * 0.95));
complete(:,1:4) = complete(:,1:4) / max(max(complete(:,1:4) * 0.95));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% permute the data
% set the number of folds (this is a parameter)

% Generate the folds
% collect the total size of the dataset
InSize = com_r;

% combine the test and training data and permute them
Perm = randsample(1:InSize,InSize);

% set the index of the test and training data
min_val = 1;
tempo = zeros(com_r,1);
for i = 1:FoldCount
    temp_ind = Perm(min_val:(com_r/FoldCount)*i);
    tempo(temp_ind) = i;
    min_val = (com_r/FoldCount)*i + 1;
end

complete = [complete tempo];
complete_res = [complete_res tempo];

%INVEC CONTAINS ALL 1-ELEMENT INPUTS IN ONE VECTOR:
%invec(1): num_pe_2
%invec(2): K
%invec(3): nu
%invec(4): maxit
%invec(5): epsilon
%invec(6): mom
%invec(7): b
%invec(8): num_plots
%invec(9): scale_val
%invec(10): scale_val_test
%% Function call
invec = [10 1 0.01 2000000 0.1 0.3 1 50 1 1]; %samp_size = 21000, half = 0
plot_cell = {}; %All of these cells are to be used for plotting and interpretation
res_cell = {};
class_cell_train = {};
class_cell_test = {};
D_train_cell = {};
D_test_cell = {};
x_train_cell = {};
x_test_cell = {};
%out_cell = cell(1,3);
for j = 1:FoldCount
    if FoldCount == 1
        x_train = train(:,1:4);
        D_train = train_res;
        x_test = test(:,1:4);
        D_test = test_res;
    else
        x_train = complete(find(complete(:,end) ~= j),1:4); %Get new fold for each iteration
        D_train = complete_res(find(complete_res(:,end) ~= j),1:end-1);
        x_test = complete(find(complete(:,end) == j),1:4);
        D_test = complete_res(find(complete_res(:,end) == j),1:end-1);
    end
    [foldr,~] = size(x_train);
    if invec(2) > foldr; invec(2) = foldr; end
    [~,plot_vals,w,v,i] = bplearn(x_train, x_test, D_train, D_test, invec);
    
    [er_fin,out_fin] = bprecall(w,v,x_train,D_train,1); %Run recall on the outputted
    %[~,total_er_fin] = thresh_er(er_fin, D_train);
    [er_fin_test,out_test_fin] = bprecall(w,v,x_test,D_test,1);
    %[test_thresh,total_er_fin_test] = thresh_er(er_fin_test, D_test);
    
    %out_cell{j} = out_vals;
    %thresh_er_test = (D_test - test_thresh);
    %res_cell{j} = thresh_er_test;
    
    class_cell_train{j} = class(out_fin);
    class_cell_test{j} = class(out_test_fin);
    
    final_error_train = 1 - classerror(class_cell_train{j}, D_train);
    final_error_test = 1 - classerror(class_cell_test{j}, D_test);
    
    plot_vals(1,end + 1) = final_error_train;
    plot_vals(2,end) = i;
    plot_vals(3,end) = final_error_test;
    plot_cell{j} = plot_vals;
    disp(['Fold ' num2str(j) ' iter: ' num2str(i)])
    
    plot_cell{j} = plot_vals;
    D_train_cell{j} = D_train;
    D_test_cell{j} = D_test;
    x_train_cell{j} = x_train;
    x_test_cell{j} = x_test;
    %% CHECKING IF THEY ALL THRESHOLD TO SAME VALUE
    disp('sum col 1 train'); disp(sum(class_cell_train{j}(:,1)) / length(class_cell_train{j}))
    disp('sum col 1 D'); disp(sum(D_train(:,1)) / length(D_train));
end

class_mat_train = {zeros(2,2),zeros(2,2),zeros(2,2)};
class_mat_test = {zeros(2,2),zeros(2,2),zeros(2,2)};
class_acc_train = zeros(1,2);
class_acc_test = zeros(1,2);
tr_class_cell = cell(1,2);
tst_class_cell = cell(1,2);
for i = 1:length(class_cell_train)
    for class_i = 1:length(tr_class_cell)
        tr_class_cell{class_i} = find(D_train_cell{i}(:,class_i) == 1);
        tst_class_cell{class_i} = find(D_test_cell{i}(:,class_i) == 1);
        class_mat_train{i}(class_i,:) = sum(class_cell_train{i}(tr_class_cell{class_i},:));
        class_mat_test{i}(class_i,:) = sum(class_cell_test{i}(tst_class_cell{class_i},:));
    end
    class_acc_train(i) = sum(diag(class_mat_train{i})) / sum(sum(class_mat_train{i}));
    class_acc_test(i) = sum(diag(class_mat_test{i})) / sum(sum(class_mat_test{i}));
end

%% Outputs
for m = 1:length(plot_cell)
    figure
    hold on
    plot(plot_cell{m}(2,:),plot_cell{m}(1,:)) %train
    plot(plot_cell{m}(2,:),plot_cell{m}(3,:), 'k') %test
    xlabel('Learn steps')
    ylabel('% mismatched')
    title_str = ['Error by learn steps for fold ' num2str(m)];
    title(title_str)
    legend('Training data', 'Test data')
    hold off
    
    disp(['Train confusion matrix for fold ' num2str(m)])
    disp(class_mat_train{m})
    disp(['Train classification acc for fold ' num2str(m) ': ' num2str(class_acc_train(m))])
    
    disp(['Test confusion matrix for fold ' num2str(m)])
    disp(class_mat_test{m})
    disp(['Test classification acc for fold ' num2str(m) ': ' num2str(class_acc_test(m))])
    
end
%% Learn Function
    function [out_vals,plot_vals,w,v,i] = bplearn(x, x_test, D, D_test, invec)
        %Returns the out_vals and plot_vals, both used for plotting, and w,v, and i
        [r, c] = size(x);
        [~,num_pe_3] = size(D);
        bias = 1;
        w_init = 0.1; %size of initial weights
        w = -w_init + (w_init - -w_init)* rand(invec(1),c + 1);
        v = -w_init + (w_init - -w_init)* rand(num_pe_3,invec(1) + 1);
        %disp('Initial W ='); disp(w)
        %disp('Initial V ='); disp(v)
        i = 1; %iterator
        cnt = 1;
        total_error = inf; %set high to avoid triggering limit
        batch_vars = 1:r; %number of patterns
        vals = datasample(batch_vars, invec(2), 'Replace', false); %random subset
        v_change = zeros(size(v)); %Initialize change matrices
        w_change = zeros(size(w));
        v_change_prev = v_change; %Initialize momentum change matrices
        w_change_prev = w_change;
        plot_vals = [];
        out_vals = [];
        while i < invec(4) && total_error > invec(5)
            for batchiter = 1:length(vals)
                k = vals(batchiter); %go through the batch variables in order
                x0 = [bias x(k,:)]'; %initial values fed to the network
                Net_hid = w * x0;
                y_1 = [bias tanh(invec(7) * Net_hid)']'; %output from hidden layer
                Net_out = v * y_1;
                y_out = tanh(invec(7) * Net_out); %final output from network
                delta_out = (D(k,:)' - y_out)' .* (1 - y_out.^2)'; %output delta
                delta_hid = delta_out * v .* (1 - y_1.^2)'; %hidden deltas
                delta_hid(1) = []; %delete bias delta
                v_change = v_change + (invec(3) * delta_out' * y_1'); %update the V
                %change matrix for this batch
                w_change = w_change + (invec(3) * delta_hid' * x0'); %update the W
            end
            v = v + v_change + (invec(6) * v_change_prev); %update V matrix (includes momentum term)
            w = w + w_change + (invec(6) * w_change_prev); %update W matrix
            vals = datasample(batch_vars, invec(2), 'Replace', false);%reset
            v_change_prev = v_change; %old W change becomes current momentum change matrix
            w_change_prev = w_change;
            v_change = zeros(size(v)); %reset V
            w_change = zeros(size(w)); %reset W
            
            %[error,~] = bprecall(w,v,x,D,bias);
            
            %[~,total_error] = thresh_er(error, D);
            
            if mod(i, invec(4)/invec(8)) == 0
                
                [err,output] = bprecall(w,v,x,D,bias); %test on W,V matrix
                %[~,total_error] = thresh_er(err, D);
                [er_test,test_raw_output] = bprecall(w,v,x_test,D_test,bias);
                %[~,total_er_test] = thresh_er(er_test, D_test);
                
                
                train_out = class(output);
                test_out = class(test_raw_output);
                total_error = 1 - classerror(train_out, D);
                total_er_test = 1 - classerror(test_out, D_test);
                
                
                
                clc
                disp(cnt * 100 / invec(8)); disp(total_error);
                
                %out_vals(:,cnt) = output * invec(9);
                plot_vals(1,cnt) = total_error;
                plot_vals(2,cnt) = i;
                plot_vals(3,cnt) = total_er_test;
                
                cnt = cnt + 1;
            end
            i = i + 1;
        end
    end
%% Functions
    function [er, out] = bprecall(w,v,x,D,bias)
        %Returns the total error matrix, which can be used to find RMSE, and the output vector
        bias_term_x = ones(length(x),1) * bias;
        y_1 = tanh(w * [bias_term_x x]')';
        [r,~] = size(y_1);
        bias_term_y = ones(r,1) * bias;
        y_out = tanh(v * [bias_term_y y_1]')';
        er = D - y_out;
        out = y_out;
    end
    function [th, er] = thresh_er(in, D)
        [r,c] = size(in);
        out_temp = zeros(r,c);
        er_sigma = 0;
        for indr = 1:r
            for indc = 1:c
                if abs(in(indr,indc)) <= 0.25
                    out_temp(indr,indc) = D(indr,indc);
                else
                    out_temp(indr,indc) = in(indr,indc);
                    er_sigma = er_sigma + 1;
                end
            end
        end
        er = er_sigma / (r * c);
        th = out_temp;
    end
    function [out] = class(in)
        [r,c] = size(in);
        out = zeros(r,c);
        for ii = 1:r
            temp = in(ii,:);
            where = find(temp == max(temp));
            out(ii,where) = 1;
        end
    end
    function [er] = classerror(in, D)
        [r,c] = size(in);
        cond = in(:,1) == D(:,1);
        temp = sum(cond);
        er = temp / r;
    end
end