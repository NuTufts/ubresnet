import os,sys

# SegData: class to hold batch data
# we expect LArCV1Dataset to fill this object
class SegData:
    def __init__(self):
        self.dim = None
        self.images = None # adc image
        self.labels = None # labels
        self.weights = None # weights
        return

    def shape(self):
        if self.dim is None:
            raise ValueError("SegData instance hasn't been filled yet")
        return self.dim

# Data interface
class LArCV1Dataset:
    def __init__(self, name, cfgfile ):
        # inputs
        # cfgfile: path to configuration. see test.py.ipynb for example of configuration
        self.name = name
        self.cfgfile = cfgfile
        return
      
    def init(self):
        # create instance of data file interface
        self.io = larcv.ThreadDatumFiller(self.name)
        self.io.configure(self.cfgfile)
        self.nentries = self.io.get_n_entries()
        self.io.set_next_index(0)
        print "[LArCV1Data] able to create ThreadDatumFiller"
        return
        
    def getbatch(self, batchsize):
        self.io.batch_process(batchsize)
        time.sleep(0.1)
        itry = 0
        while self.io.thread_running() and itry<100:
            time.sleep(0.01)
            itry += 1
        if itry>=100:
            raise RuntimeError("Batch Loader timed out")
        
        # fill SegData object
        data = SegData()
        dimv = self.io.dim() # c++ std vector through ROOT bindings
        self.dim     = (dimv[0], dimv[1], dimv[2], dimv[3] )
        self.dim3    = (dimv[0], dimv[2], dimv[3] )

        # numpy arrays
        data.np_images  = np.zeros( self.dim,  dtype=np.float32 )
        data.np_labels  = np.zeros( self.dim3, dtype=np.int )
        data.np_weights = np.zeros( self.dim3, dtype=np.float32 )
        data.np_images[:]  = larcv.as_ndarray(self.io.data()).reshape(    self.dim  )[:]
        data.np_labels[:]  = larcv.as_ndarray(self.io.labels()).reshape(  self.dim3 )[:]
        data.np_weights[:] = larcv.as_ndarray(self.io.weights()).reshape( self.dim3 )[:]
        data.np_labels[:] += -1
        
        # pytorch tensors
        data.images = torch.from_numpy(data.np_images)
        data.labels = torch.from_numpy(data.np_labels)
        data.weight = torch.from_numpy(data.np_weights)
        
        return data
