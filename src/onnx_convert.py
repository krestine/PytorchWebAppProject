import torch.onnx
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict

class SimpleCNN(nn.Module):

    #Our batch shape for input x is (1, 28, 28)

    def __init__(self, hidden_size = 64, output_size = 10):
        '''
        Init method
        INPUT:
            hidden_size - size of the hidden fully-connnected layer
            output_size - size of the output
        '''
        super(SimpleCNN, self).__init__()

        #Input channels = 3, output channels = 18
        self.conv1 = nn.Conv2d(1, 18, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        #3528 input features, 64 output features (see sizing flow below)
        self.fc1 = nn.Linear(18 * 14 * 14, hidden_size)

        #64 input features, 10 output features for our 10 defined classes
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        '''
        Forward pass of the model.
        INPUT:
            x - input data
        '''

        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))

        #Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)

        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 16, 16) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 18 * 14 * 14)

        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))

        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return(x)

#Function to Convert to ONNX
def Convert_ONNX(input_size):

    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, input_size, requires_grad=True)

    # Export the model
    torch.onnx.export(model,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         "ImageClassifier.onnx",       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         opset_version=10,    # the ONNX version to export the model to
         do_constant_folding=True,  # whether to execute constant folding for optimization
         input_names = ['modelInput'],   # the model's input names
         output_names = ['modelOutput'], # the model's output names
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
                                'modelOutput' : {0 : 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')


if __name__ == "__main__":
    # Let's build our model
    # train(5)
    # print('Finished Training')

    # Test which classes performed well
    # testAccuracy()
    input_size = 784
    hidden_sizes = [128, 100, 64]
    output_size = 10
    dropout = 0.0

    # Let's load the model we just created and test the accuracy per label
    model = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_sizes[0])),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        ('bn2', nn.BatchNorm1d(num_features=hidden_sizes[1])),
        ('relu2', nn.ReLU()),
        ('dropout', nn.Dropout(dropout)),
        ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
        ('bn3', nn.BatchNorm1d(num_features=hidden_sizes[2])),
        ('relu3', nn.ReLU()),
        ('logits', nn.Linear(hidden_sizes[2], output_size))]))
    # model = SimpleCNN(64, output_size)

    path = "./checkpoint.pth"
    loaded_model = torch.load(path)
    model.load_state_dict(loaded_model)

    # Test with batch of images
    # testBatch()
    # Test how the classes performed
    # testClassess()

    # Conversion to ONNX
    Convert_ONNX(input_size=input_size)