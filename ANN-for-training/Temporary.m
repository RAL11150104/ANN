% load RESULTS_14_14.mat;
% load RESULTS_15_15.mat;

%% PLOTTING 13 13 %%

load RESULTS_13_13.mat;

figure(1)
epoch = [500,1000,1500,2000,2500];
plot(epoch,final_test_accuracy(1,:),'Marker','o','linewidth',1);
xlabel('Epoch')
ylabel('Accuracy')
title('Graph of Epoch vs Test Accuracy (13 Hidden Nodes)')
hold on;
plot(epoch,final_test_accuracy(2,:),'Marker','o','linewidth',1);
plot(epoch,final_test_accuracy(3,:),'Marker','o','linewidth',1);
plot(epoch,final_test_accuracy(4,:),'Marker','o','linewidth',1);
plot(epoch,final_test_accuracy(5,:),'Marker','o','linewidth',1);
x = legend('1e-1','1e-2','1e-3','1e-4','1e-5','Location','Southeast');
title(x,'Learning Rate')
hold off;

figure(2)
plot(epoch,final_train_accuracy(1,:),'Marker','*','linewidth',1);
hold on;
plot(epoch,final_train_accuracy(2,:),'Marker','*','linewidth',1);
plot(epoch,final_train_accuracy(3,:),'Marker','*','linewidth',1);
plot(epoch,final_train_accuracy(4,:),'Marker','*','linewidth',1);
plot(epoch,final_train_accuracy(5,:),'Marker','*','linewidth',1);
xlabel('Epoch')
ylabel('Accuracy')
title('Graph of Epoch vs Train Accuracy (13 Hidden Nodes)')
x = legend('1e-1','1e-2','1e-3','1e-4','1e-5','Location','Southeast');
title(x,'Learning Rate')
hold off;

%% PLOTTING 14 14 %%

load RESULTS_14_14.mat;

figure(1)
epoch = [500,1000,1500,2000,2500];
plot(epoch,final_test_accuracy(1,:),'Marker','o','linewidth',1);
xlabel('Epoch')
ylabel('Accuracy')
title('Graph of Epoch vs Testing Accuracy (14 Hidden Nodes)')
hold on;
plot(epoch,final_test_accuracy(2,:),'Marker','o','linewidth',1);
plot(epoch,final_test_accuracy(3,:),'Marker','o','linewidth',1);
plot(epoch,final_test_accuracy(4,:),'Marker','o','linewidth',1);
plot(epoch,final_test_accuracy(5,:),'Marker','o','linewidth',1);
x = legend('1e-1','1e-2','1e-3','1e-4','1e-5','Location','Southeast');
title(x,'Learning Rate')
hold off;

figure(2)
plot(epoch,final_train_accuracy(1,:),'Marker','*','linewidth',1);
hold on;
plot(epoch,final_train_accuracy(2,:),'Marker','*','linewidth',1);
plot(epoch,final_train_accuracy(3,:),'Marker','*','linewidth',1);
plot(epoch,final_train_accuracy(4,:),'Marker','*','linewidth',1);
plot(epoch,final_train_accuracy(5,:),'Marker','*','linewidth',1);
xlabel('Epoch')
ylabel('Accuracy')
title('Graph of Epoch vs Training Accuracy (14 Hidden Nodes)')
x = legend('1e-1','1e-2','1e-3','1e-4','1e-5','Location','Southeast');
title(x,'Learning Rate')
hold off;

%% PLOTTING 13 13 %%

load RESULTS_15_15.mat;

figure(1)
epoch = [500,1000,1500,2000,2500];
plot(epoch,final_test_accuracy(1,:),'Marker','o','linewidth',1);
xlabel('Epoch')
ylabel('Accuracy')
title('Graph of Epoch vs Testing Accuracy (15 Hidden Nodes)')
hold on;
plot(epoch,final_test_accuracy(2,:),'Marker','o','linewidth',1);
plot(epoch,final_test_accuracy(3,:),'Marker','o','linewidth',1);
plot(epoch,final_test_accuracy(4,:),'Marker','o','linewidth',1);
plot(epoch,final_test_accuracy(5,:),'Marker','o','linewidth',1);
x = legend('1e-1','1e-2','1e-3','1e-4','1e-5','Location','Southeast');
title(x,'Learning Rate')
hold off;

figure(2)
plot(epoch,final_train_accuracy(1,:),'Marker','*','linewidth',1);
hold on;
plot(epoch,final_train_accuracy(2,:),'Marker','*','linewidth',1);
plot(epoch,final_train_accuracy(3,:),'Marker','*','linewidth',1);
plot(epoch,final_train_accuracy(4,:),'Marker','*','linewidth',1);
plot(epoch,final_train_accuracy(5,:),'Marker','*','linewidth',1);
xlabel('Epoch')
ylabel('Accuracy')
title('Graph of Epoch vs Training Accuracy (15 Hidden Nodes)')
x = legend('1e-1','1e-2','1e-3','1e-4','1e-5','Location','Southeast');
title(x,'Learning Rate')
hold off;