import numpy as np
import scipy.io
import os
import itertools
import glob
import PIL.Image as Image

class dataloader:

    def __init__(self, batch_size=1024, shuffle=True, augmentations={}, from_disk=False, path_to_data=None):

        # Parameters for the dataloader
        self.batch_size = batch_size #what is the batch size
        self.shuffle = shuffle #shuffle or not
        self.augmentations = augmentations #list of augmentations
        self.from_disk = from_disk #load from disk?
        if self.from_disk == True and path_to_data==None:
            raise Exception("Input Error. If loading from disk, path_to_data must not be None")
        
        self.path_to_data = path_to_data        
        if self.from_disk:
            self.num_examples = dataloader.scan_data_path(path_to_data)

    def __len__(self):
        if self.from_disk:
            return len(self.run_sheet)
        else:
            try:
                return self.X.shape[0]
            except:
                return len(self.run_sheet)
            finally: #is this error handling right?
                print("Dataset not loaded yet")


    def printSettings(self):
        """
        Prints data_loader settings to console
        """
        print(f'Batch Size: {self.batch_size}')
        print(f'Shuffle: {self.shuffle}')
        print(f'Augmentations: {self.augmentations}')
        print(f'Load From Disk?: {self.from_disk}')
        if self.from_disk:
            print(f'Path to Data: {self.path_to_data}')



    def load_svhn(self, path_to_data):
        """
        svhn specific loader
        """
        training_data = scipy.io.loadmat(file_name=path_to_data)
        training_data['X'] = np.transpose(training_data['X'], axes=(3,0,1,2)) # NHWC
        
        #replace label y==10 --> 0 convention
        training_data['y'][training_data['y']==10] = 0

        if training_data['X'].shape[0] != training_data['y'].shape[0]:
            raise Exception("Dimension Mismatch. X and y do not have the same number of samples. Check dimensional order")

        self.X,self.y = training_data['X'], training_data['y']

        self.num_examples = self.X.shape[0] #assume N is first dim
        self.path_to_data = path_to_data


    def load_direct(self,X,y): 
        """
        Directly provide X and y and load into memory. expects batch dim as first
        """
        if X.shape[0] != y.shape[0]:
            raise Exception("Dimension Mismatch. X and y do not have the same number of samples")

        print(f'Sucessfully loaded {y.shape[0]} examples and labels')

        self.X = X
        self.y = y
        self.num_examples = X.shape[0]


    def generate_batches(self):
        """
        Generate a batching 'run-sheet'.
        It is a list containing example indices, in batches of batch_size.
        """   
        run_sheet = list(range(0,self.num_examples))
        
        #shuffle the sequence if requested
        if self.shuffle:
            np.random.shuffle(run_sheet)
        
        self.run_sheet = list(dataloader.grouper(iterable=run_sheet, batch_size=self.batch_size))

    def get_batch(self):
        """
        yield a batch of data on call
        """
        for batch in self.run_sheet:
            if self.from_disk == True:
            # load the files given by the index numbers in the batch and append them together
                batch_X, batch_y = self.load_from_disk(batch)
            else:
                batch_X = self.X[batch,...]
                batch_y = self.y[batch,...]

            batch_X = dataloader.apply_augmentations(batch_X, self.augmentations)

            yield batch_X, batch_y



    def load_from_disk(self, batch):
        """
        Load the file directly from disk, read label from filename
        """
        # item in this case should be an example-number from the dataset which we want to pass on after loading.
        batch_X = []
        batch_y = []

        for item in batch:
            # find the example in the input_data folder.
            # this would need work to make it fail-proof but if we are creating the filenames..should be OK
            for name in glob.glob(os.path.join(self.path_to_data, f'img_{item}_*.png')):

                # load example
                X = Image.open(name)
                X = np.array(X.getdata()).reshape(X.size[0],X.size[1],3)
                batch_X.append(X)
                
                # label is encoded in filename so separate it.
                #  first chop the extension off
                fn = name.split(sep='.')[0]
                # grab the label now from the end.
                y = fn.split(sep='_')[-1]

                batch_y.append(y)

        batch_X = np.transpose(np.array(batch_X),axes=(0,3,1,2)) # re-orient to NCHW
        batch_y[batch_y==10] = 0

        return batch_X, batch_y

    def scan_data_path(path_to_data):

        image_list = glob.glob(os.path.join(path_to_data, f'img_*_*.png'))

        num_examples = len(image_list)

        return num_examples


    def apply_augmentations(batch, composed_augmentations):
        """
        Apply Augmentations that pass the probability threshold to the data
        """
        output = []
        for ex in range(0, batch.shape[0]):
            output.append(composed_augmentations(batch[ex,...]))

        return output


    def grouper(iterable, batch_size, fillvalue=0):
        """
        Generate a batching 'run-sheet' using itertools recipe for grouping. 
        Default fill any non-full groups with 0
        """

        # Collect data into non-overlapping fixed-length chunks or blocks
        # grouper('ABCDEFG', 3, fillvalue='x') --> ABC DEF Gxx
        args = [iter(iterable)] * batch_size

        return itertools.zip_longest(*args, fillvalue=fillvalue)
