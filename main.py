from model import *
from generate_data import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def main():
    # generate_data
    dataset_train,dataset_test = generate_data()
    #neural network
    model = neural_network()
    train(model,dataset_train)
    test(model,dataset_test)

def train(model,dataset_train):
    #dataloader
    train_loader = DataLoader(dataset_train,shuffle=True,batch_size=256)
    # optimizer , loss
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
    epochs = 1500
    #training loop
    for i in range(epochs):
        for j,data in enumerate(train_loader):
            x = data[:][0] # batch * time (256 * 10)
            x = x.view(-1,10,1) # batch * time * input_size (256 * 10 * 1)
            y_pred = model(x) # batch * output_size (256 * 1)
            y_pred = y_pred.view(-1) # batch  (256)
            loss = criterion(y_pred,data[:][1])
            loss.backward()
            optimizer.step()
        if i%100 == 0:
            print(i,"th iteration : ",loss)

def test(model,dataset_test):
    #test set actual vs predicted
    test_pred = model(dataset_test[:][0].view(-1,10,1)).view(-1)
    plt.figure()
    plt.plot(test_pred.detach().numpy(),label='predicted')
    plt.plot(dataset_test[:][1].view(-1),label='original')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()