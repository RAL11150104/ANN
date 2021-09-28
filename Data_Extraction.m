%% DATA EXTRACTION PART OF MNIST DATASET %%

% FUNCTION - extracting data from .csv file %
% OUTPUT: .mat file of 100 samples for classes 0,1 &2 %

% NOTES %
% Randomized for removing bias %

%% EXTRACTION %%

mnist_train = readtable('mnist_train.csv');
mnist_test = readtable('mnist_test.csv');

sample_mnist_train = table2array(sortrows(mnist_train,1));
sample_mnist_test = table2array(sortrows(mnist_test,1));

train_class_zero = sample_mnist_train((sample_mnist_train(:,1) == 0),:);
train_class_one = sample_mnist_train((sample_mnist_train(:,1) == 1),:);
train_class_two = sample_mnist_train((sample_mnist_train(:,1) == 2),:);

test_class_zero = sample_mnist_test((sample_mnist_test(:,1) == 0),:);
test_class_one = sample_mnist_test((sample_mnist_test(:,1) == 1),:);
test_class_two = sample_mnist_test((sample_mnist_test(:,1) == 2),:);

rng(1); % For Reproducibility

train_index_zero = randperm(size(train_class_zero,1),100);
train_index_one = randperm(size(train_class_one,1),100);
train_index_two = randperm(size(train_class_two,1),100);

test_index_zero = randperm(size(test_class_zero,1),100);
test_index_one = randperm(size(test_class_one,1),100);
test_index_two = randperm(size(test_class_two,1),100);

temporary_train_mnist_dataset = [train_class_zero(train_index_zero(1:100),:);train_class_one(train_index_one(1:100),:);train_class_two(train_index_two(1:100),:)];
temporary_test_mnist_dataset = [test_class_zero(test_index_zero(101:end),:);test_class_one(test_index_one(101:end),:);test_class_two(test_index_two(101:end),:)];

save temporary_train_mnist_dataset.mat temporary_train_mnist_dataset;
save temporary_test_mnist_dataset.mat temporary_test_mnist_dataset;
