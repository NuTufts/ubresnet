# builtins
import os,sys,time
from collections import OrderedDict
import argparse

# numpy
import numpy as np

# ROOT/larcv
import ROOT as rt
from ROOT import TH1F,TTree,TFile
from larcv import larcv

# pytorch
import torch

# util functions
# also, implicitly loads dependencies, pytorch larflow model definition
from infill_funcs import load_model

# larflow
if "LARCVDATASET_BASEDIR" in os.environ:
    sys.path.append(os.environ["LARCVDATASET_BASEDIR"])
else:
    sys.path.append("../pytorch-larflow/larcvdataset") # default location
from larcvdataset import LArCVDataset

def load_pre_cropped_data( larcvdataset_configfile, batchsize=1 ):
    larcvdataset_config="""ThreadProcessor: {
        Verbosity:3
        NumThreads: 2
        NumBatchStorage: 2
        RandomAccess: false
        InputFiles: ["../training/inputfiles/cropinfillADC_test.root"]
        ProcessName: ["ADC_valid","ADCmasked_valid","weights_valid","labelsbasic_valid"]
        ProcessType: ["BatchFillerImage2D","BatchFillerImage2D","BatchFillerImage2D","BatchFillerImage2D"]
        ProcessList: {
            weights_valid: {
                Verbosity:3
                ImageProducer: "Weights"
                Channels: [2]
                EnableMirror: false
            }
            ADC_valid: {
                Verbosity:3
                ImageProducer: "ADC"
                Channels: [2]
                EnableMirror: false
            }
            labelsbasic_valid: {
                Verbosity:3
                ImageProducer: "LabelsBasic"
                Channels: [2]
                EnableMirror: false
            }
            ADCmasked_valid: {
                Verbosity:3
                ImageProducer: "ADCMasked"
                Channels: [2]
                EnableMirror: false
            }
        }
    }

    """
    with open("larcv_dataloader.cfg",'w') as f:
        print >> f,larcvdataset_config
    iotest = LArCVDataset( "larcv_dataloader.cfg","ThreadProcessor") #, store_eventids=True

    return iotest

if __name__=="__main__":
    # ARGUMENTS DEFINTION/PARSER
    if len(sys.argv)>1:
        crop_view_parser = argparse.ArgumentParser(description='Process cropped-image views through LArFlow.')
        crop_view_parser.add_argument( "-i", "--input",        required=True, type=str, help="location of input larcv file" )
        crop_view_parser.add_argument( "-o", "--output",       required=True, type=str, help="location of output larcv file" )
        crop_view_parser.add_argument( "-c", "--checkpoint",   required=True, type=str, help="location of model checkpoint file")
        crop_view_parser.add_argument( "-g", "--gpuid",        default=0,     type=int, help="GPUID to run on")
        crop_view_parser.add_argument( "-p", "--chkpt-gpuid",  default=0,     type=int, help="GPUID used in checkpoint")
        crop_view_parser.add_argument( "-b", "--batchsize",    default=2,     type=int, help="batch size" )
        crop_view_parser.add_argument( "-v", "--verbose",      action="store_true",     help="verbose output")
        crop_view_parser.add_argument( "-v", "--nevents",      default=-1,    type=int, help="process number of events (-1=all)")

        args = crop_view_parser.parse_args(sys.argv)
        input_larcv_filename  = args.input
        output_larcv_filename = args.output
        checkpoint_data       = args.checkpoint
        gpuid                 = args.gpuid
        checkpoint_gpuid      = args.chkpt_gpuid
        batch_size            = args.batchsize
        verbose               = args.verbose
        nprocess_events       = args.nevents
    else:
        # for testing
        # for checkpoint files see: /mnt/disk1/nutufts/kmason/ubresnet/training/ on nudot
        input_larcv_filename = "../training/inputfiles/crop_valid.root" # test cropped image file
        output_larcv_filename = "output_infill.root"
        checkpoint_data = "/mnt/disk1/nutufts/kmason/ubresnet/training/test.tar"
        batch_size = 1
        gpuid = 1
        checkpoint_gpuid = 0
        verbose = True
        nprocess_events = 10

    # load data
    inputdata = load_pre_cropped_data( input_larcv_filename, batchsize=batch_size )
    inputmeta = larcv.IOManager(larcv.IOManager.kREAD )
    inputmeta.add_in_file( input_larcv_filename )
    inputmeta.initialize()
    width=512
    height=832

    # load model
    model = load_model( checkpoint_data, gpuid=gpuid, checkpointgpu=checkpoint_gpuid )
    model.to(device=torch.device("cuda:%d"%(gpuid)))
    model.eval()

    # output IOManager
    outputdata = larcv.IOManager( larcv.IOManager.kWRITE )
    outputdata.set_out_file( output_larcv_filename )
    outputdata.initialize()

    inputdata.start(batch_size)

    nevts = len(inputdata)
    if nprocess_events>=0:
        nevts = nprocess_events
    nbatches = nevts/batch_size
    if nevts%batch_size!=0:
        nbatches += 1

    # root hist to inspect diff
    f = TFile( 'test.root', 'recreate' )
    h = TH1F( 'h1' , 'diff', 150, -25., 25.)
    t = TTree('t1', 'diff histos')

    ientry = 0
    for ibatch in range(nbatches):

        if verbose:
            print "=== [BATCH %d] ==="%(ibatch)
        #
        data = inputdata[0]
        # diff_h= array( 'diff', [ 0 ] )
        t.Branch( 'diff', h, 'diff' )


        # get input ADC(masked) images
        ADCmasked_t = torch.from_numpy( data["ADCmasked_valid"].reshape( (batch_size,1,width,height) ) ) # source image ADC
        ADCmasked_t = ADCmasked_t.to(device=torch.device("cuda:%d"%(gpuid)))

        # get ADC images for accuracy calculation
        ADC_t = torch.from_numpy( data["ADC_valid"].reshape( (batch_size,1,width,height) ) )
        ADC_t = ADC_t.to(device=torch.device("cuda:%d"%(gpuid)))

        #get labels images
        labelbasic_t = torch.from_numpy( data["labelsbasic_valid"].reshape( (batch_size,1,width,height) ) )
        labelbasic_t = labelbasic_t.to(device=torch.device("cuda:%d"%(gpuid)))

        #get weights images
        weights_t = torch.from_numpy( data["weights_valid"].reshape( (batch_size,1,width,height) ) )
        weights_t = weights_t.to(device=torch.device("cuda:%d"%(gpuid)))

        #save a copy of ADC masked
        ev_out_ADCmasked = outputdata.get_data("image2d", "ADCMasked")
        ADCmaskedout_t = ADCmasked_t.detach().cpu().numpy()

        # save a copy of ADC
        ev_out_ADC = outputdata.get_data("image2d","ADC")
        ADCout_t = ADC_t.detach().cpu().numpy()

        #save a copy of labels_basic
        ev_out_labels = outputdata.get_data("image2d","LabelsBasic")
        labelsout_t = labelbasic_t.detach().cpu().numpy()

        #save a copy for use in creating an accuracy image
        acc_t = torch.from_numpy( data["labelsbasic_valid"].reshape( (batch_size,1,width,height) ) )


        # run model
        pred_ADCvalue = model.forward( ADCmasked_t) #ADC_t
        # get predictions from gpu
        ADCvalue_np = pred_ADCvalue.detach().cpu().numpy().astype(np.float32)

        for ib in range(batch_size):
            if ientry>=nevts:
                # skip last portion of last batch
                break

            # get meta
            inputmeta.read_entry(ientry)
            ev_meta   = inputmeta.get_data("image2d","ADC")
            outmeta   = ev_meta.image2d_array()[2].meta()

            # save inputs to network for reference
            # and save original images
            ADCmasked_slice=ADCmaskedout_t[ib,0,:,:]
            ADCmasked_out = larcv.as_image2d_meta(ADCmasked_slice,outmeta)
            ev_out_ADCmasked.append( ADCmasked_out )

            ADC_slice=ADCout_t[ib,0,:,:]
            ADC_out = larcv.as_image2d_meta(ADC_slice,outmeta)
            ev_out_ADC.append( ADC_out )

            labels_slice=labelsout_t[ib,0,:,:]
            labels_out = larcv.as_image2d_meta(labels_slice,outmeta)
            ev_out_labels.append( labels_out )

            # save output of network as images
            img_slice0 = ADCvalue_np[ib,0,:,:]
            out_lcv  = larcv.as_image2d_meta( img_slice0, outmeta )
            ev_out_out    = outputdata.get_data("image2d","out")
            ev_out_out.append( out_lcv )

            #create an accuracy image
            ev_out_acc = outputdata.get_data("image2d","acc")

            #variables for accuracy check
            within2= 0.0
            within5= 0.0
            within10= 0.0
            within20= 0.0
            chargetotal = 0.0


            #loop through all pixels
            # calculate accuracies
            # create acc image
            chargetotal = 0.0
            fracdead = labelbasic_t.sum().item()
            for rows in range(512):
                for cols in range(832):
                    truepix = ADC_t[ib,0,rows,cols].item()
                    outpix = ADCvalue_np[ib,0,rows,cols].item()
                    weightpix = weights_t[ib,0,rows,cols].item()
                    diff = abs(truepix - outpix)
                    if labelbasic_t[ib,0,rows,cols] == 1.0:
                        # total = total + 1
                        if diff < 2.0 and truepix != 0:
                            within2 = within2 + 1
                        if diff < 5.0 and truepix != 0:
                            within5 = within5 + 1
                        if diff < 10.0 and truepix != 0:
                            within10 = within10 + 1
                        if diff < 20.0 and truepix != 0:
                            within20 = within20 + 1
                        # ADCmasked_t[ib,0,rows,cols] = outpix
                        # ADC_t[ib,0,rows,cols] = outpix
                    #     acc_t[ib,0,rows,cols] = abs((truepix.item() * weightpix.item()) - (outpix.item()*weightpix.item()))/fracdead
                    # else:
                    #     acc_t[ib,0,rows,cols] = abs((truepix.item() * weightpix.item()) - (outpix.item()*weightpix.item()))
                        if truepix != 0:
                            h.Fill(truepix - outpix)
                            chargetotal +=1.0


                    acc_t[ib,0,rows,cols] = diff


            accuracy2 = (within2/chargetotal)*100
            accuracy5 = (within5/chargetotal)*100
            accuracy10 = (within10/chargetotal)*100
            accuracy20 = (within20/chargetotal)*100
            print "Accuracy < 2: ", accuracy2
            print "Accuracy < 5: ", accuracy5
            print "Accuracy < 10: ", accuracy10
            print "Accuracy < 20: ", accuracy20
            print "  "

            acc_out = larcv.as_image2d_meta(acc_t[ib,0,:,:], outmeta)
            ev_out_acc.append(acc_out)
            t.Fill()

            outputdata.set_id( ev_meta.run(), ev_meta.subrun(), ev_meta.event() )
            outputdata.save_entry()
            ientry += 1

        #     #masked outputs
        #     #start by copying weight and ADCmasked as the starting points
        #     if SAVE_TOTAL:
        #     	ev_out_total = outputdata.get_data("image2d", "total")
        #     if SAVE_OVERLAY:
        #     	ev_out_overlay = outputdata.get_data("image2d","overlay")

        #         if SAVE_TOTAL:
        #             total_out = larcv.as_image2d_meta(ADCmasked_t[ib,0,:,:], outmeta)
        #             ev_out_total.append(total_out)
        #         if SAVE_OVERLAY:
        #             overlay_out = larcv.as_image2d_meta(ADC_t[ib,0,:,:], outmeta)
        #             ev_out_overlay.append(overlay_out)


    # stop input
    inputdata.stop()

    # save results
    outputdata.finalize()

    f.Write()
    f.Close()

    print "DONE."
