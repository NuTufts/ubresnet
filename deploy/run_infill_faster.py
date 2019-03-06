# builtins
import os,sys,time
from collections import OrderedDict
import argparse

# numpy
import numpy as np

# ROOT/larcv
import ROOT as rt
from ROOT import TH1F,TTree,TFile,TH2F,TCanvas,TLine,TAttFill,TPad
from larcv import larcv

# pytorch
import torch

# # numba
from numba import jit

# util functions
# also, implicitly loads dependencies, pytorch larflow model definition
from infill_funcs import load_model

import time

# larflow
if "LARCVDATASET_BASEDIR" in os.environ:
    sys.path.append(os.environ["LARCVDATASET_BASEDIR"])
else:
    sys.path.append("../pytorch-larflow/larcvdataset") # default location
from larcvdataset import LArCVDataset

#Flags to choose which images to save
# True ADC image
SAVE_ADC = True
# ADC image with dead channels masked output
SAVE_ADCMASKED = True
# image with dead channels labeled
SAVE_LABELS = True
# what the network outputs
SAVE_OUTPUT =True
# image of difference between output and truth
SAVE_DIFF = False
# image of that shows the difference with thresholds
SAVE_THRESHOLDS = True
# true adc with dead channels filled w/ prediction
SAVE_OVERLAY =True


# @jit
def load_pre_cropped_data( larcvdataset_configfile, batchsize=1 ):
    larcvdataset_config="""ThreadProcessor: {
        Verbosity:3
        NumThreads: 2
        NumBatchStorage: 2
        RandomAccess: false
        InputFiles: ["crop_sample.root"]
        ProcessName: ["ADC_valid","ADCmasked_valid","weights_valid","labelsbasic_valid"]
        ProcessType: ["BatchFillerImage2D","BatchFillerImage2D","BatchFillerImage2D","BatchFillerImage2D"]
        ProcessList: {
            weights_valid: {
                Verbosity:3
                ImageProducer: "Weights"
                Channels: [0]
                EnableMirror: false
            }
            ADC_valid: {
                Verbosity:3
                ImageProducer: "ADC"
                Channels: [0]
                EnableMirror: false
            }
            labelsbasic_valid: {
                Verbosity:3
                ImageProducer: "LabelsBasic"
                Channels: [0]
                EnableMirror: false
            }
            ADCmasked_valid: {
                Verbosity:3
                ImageProducer: "ADCMasked"
                Channels: [0]
                EnableMirror: false
            }
        }
    }

    """

    with open("larcv_dataloader.cfg",'w') as f:
        print >> f,larcvdataset_config
    iotest = LArCVDataset( "larcv_dataloader.cfg","ThreadProcessor") #, store_eventids=True

    return iotest


def pixelloop(within2,
    within5,
    within10,
    within20,
    chargetotal,
    labelbasic_numpy,
    weights_numpy,
    ADC_numpy,
    ADCvalue_numpy,
    overlay_numpy,
    thresh_numpy,
    h, h2):

    #loop through all pixels
    # calculate accuracies
    # create acc image
    chargetotal = 0.0
    for rows in range(512):
        for cols in range(832):
            truepix = ADC_numpy[ib,0,rows,cols].item()
            outpix = ADCvalue_numpy[ib,0,rows,cols].item()
            weightpix = weights_numpy[ib,0,rows,cols].item()
            diff = abs(truepix - outpix)

            if labelbasic_numpy[ib,0,rows,cols].item() == 1.0:
                # fill overlay_t
                overlay_numpy[ib,0,rows,cols] = outpix
                # calculate accuracies
                if diff < 2.0 and truepix > 0:
                    within2 = within2 + 1
                if diff < 5.0 and truepix > 0:
                    within5 = within5 + 1
                if diff < 10.0 and truepix > 0:
                    within10 = within10 + 1
                if diff < 20.0 and truepix > 0:
                    within20 = within20 + 1

                # make diff histogram
                if truepix > 0:
                    if truepix < 10:
                        truepix=0
                    if outpix < 10:
                        outpix=0
                    h.Fill(truepix - outpix)
                    h2.Fill(truepix,outpix)
                    chargetotal +=1.0

            # make threshold image
            if overlay_numpy[ib,0,rows,cols] < 10:
                overlay = 0
                overlay_numpy[ib,0,rows,cols] = 0
            else:
                overlay = overlay_numpy[ib,0,rows,cols]

            if ADC_numpy[ib,0,rows,cols] < 10:
                ADC = 0
            else:
                ADC = ADC_numpy[ib,0,rows,cols]

            diffoverlay = abs(overlay-ADC)
            if diff < 2.0:
                thresh_numpy[ib,0,rows,cols] = 0
            elif diff < 5.0:
                thresh_numpy[ib,0,rows,cols] = 1
            elif diff < 10.0:
                thresh_numpy[ib,0,rows,cols] = 2
            elif diff < 20.0:
                thresh_numpy[ib,0,rows,cols] = 3
            else:
                thresh_numpy[ib,0,rows,cols] = 4


    return within2, within5, within10, within20, chargetotal,overlay_numpy, thresh_numpy, h, h2

if __name__=="__main__":
    # ARGUMENTS DEFINTION/PARSER
    start = time.time()
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
        input_larcv_filename = "crop_sample.root" # test cropped image file
        output_larcv_filename = "output_infill_GAN.root"
        checkpoint_data = "/mnt/disk1/nutufts/kmason/ubresnet/training/MCResults/uplanefromv_MC_33000.tar"
        batch_size = 1
        gpuid = 1
        checkpoint_gpuid = 0
        verbose = True
        nprocess_events = 100

    # load data
    inputdata = load_pre_cropped_data( input_larcv_filename, batchsize=batch_size )
    inputmeta = larcv.IOManager(larcv.IOManager.kREAD )
    inputmeta.add_in_file( input_larcv_filename )
    inputmeta.initialize()
    width=832
    height=512

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
    h2 = TH2F('h2' , 'diff2d', 50,0.,50.,50,0.,50.)
    t = TTree('t1', 'diff histos')

    ientry = 0
    averageacc2 = 0.0
    averageacc5 = 0.0
    averageacc10 = 0.0
    averageacc20 = 0.0
    for ibatch in range(nbatches):

        if verbose:
            print "=== [BATCH %d] ==="%(ibatch)
        #
        data = inputdata[0]
        # diff_h= array( 'diff', [ 0 ] )
        t.Branch( 'diff', h, 'diff' )


        # get input ADC(masked) images
        ADCmasked_t = torch.from_numpy( data["ADCmasked_valid"].reshape( (batch_size,1,height,width) ) ) # source image ADC
        ADCmasked_t = ADCmasked_t.to(device=torch.device("cuda:%d"%(gpuid)))

        # get ADC images for accuracy calculation
        ADC_t = torch.from_numpy( data["ADC_valid"].reshape( (batch_size,1,height,width) ) )
        #ADC_t = ADC_t.to(device=torch.device("cuda:%d"%(gpuid)))

        #get labels images
        labelbasic_t = torch.from_numpy( data["labelsbasic_valid"].reshape( (batch_size,1,height,width) ) )
        # labelbasic_t = labelbasic_t.to(device=torch.device("cuda:%d"%(gpuid)))


        #get weights images
        weights_t = torch.from_numpy( data["weights_valid"].reshape( (batch_size,1,height,width) ) )
        # weights_t = weights_t.to(device=torch.device("cuda:%d"%(gpuid)))

        #save a copy of ADC masked
        if SAVE_ADCMASKED:
            ev_out_ADCmasked = outputdata.get_data("image2d", "ADCMasked")
            ADCmaskedout_t = ADCmasked_t.detach().cpu().numpy()

        # save a copy of ADC
        if SAVE_ADC:
            ev_out_ADC = outputdata.get_data("image2d","ADC")
            ADCout_t = ADC_t.detach().cpu().numpy()

        #save a copy of labels_basic
        if SAVE_LABELS:
            ev_out_labels = outputdata.get_data("image2d","LabelsBasic")
            labelsout_t = labelbasic_t.detach().cpu().numpy()



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
            if SAVE_ADCMASKED:
                ADCmasked_slice=ADCmaskedout_t[ib,0,:,:]
                ADCmasked_out = larcv.as_image2d_meta(ADCmasked_slice,outmeta)
                ev_out_ADCmasked.append( ADCmasked_out )

            if SAVE_ADC:
                ADC_slice=ADCout_t[ib,0,:,:]
                ADC_out = larcv.as_image2d_meta(ADC_slice,outmeta)
                ev_out_ADC.append( ADC_out )

            if SAVE_LABELS:
                labels_slice=labelsout_t[ib,0,:,:]
                labels_out = larcv.as_image2d_meta(labels_slice,outmeta)
                ev_out_labels.append( labels_out )

            # save output of network as images
            img_slice0 = ADCvalue_np[ib,0,:,:]
            if SAVE_OUTPUT:
                out_lcv  = larcv.as_image2d_meta( img_slice0, outmeta )
                ev_out_out    = outputdata.get_data("image2d","out")
                ev_out_out.append( out_lcv )

            #create a thresh, diff, and overlay image
            if SAVE_DIFF:
                ev_out_diff = outputdata.get_data("image2d","diff")
            if SAVE_THRESHOLDS:
                ev_out_thresh = outputdata.get_data("image2d","thresh")
            if SAVE_OVERLAY:
                ev_out_overlay= outputdata.get_data("image2d","overlay")

            beforeloop = time.time()

            #variables for accuracy check
            within2= 0.0
            within5= 0.0
            within10= 0.0
            within20= 0.0
            chargetotal = 0.0

            labelbasic_numpy= labelbasic_t.numpy()
            weights_numpy= weights_t.numpy()
            ADC_numpy= ADC_t.numpy()

            #save a copy of labels for use in creating diff and threshold images
            thresh_t = torch.from_numpy( data["weights_valid"].reshape( (batch_size,1,height,width) ) )
            #save a copy of adc for creating overlay
            overlay_t = torch.from_numpy( data["ADCmasked_valid"].reshape( (batch_size,1,height,width) ) )

            overlay_numpy = overlay_t.numpy()
            thresh_numpy = thresh_t.numpy()

            within2, within5, within10, within20, chargetotal, overlay_numpy, thresh_numpy, h, h2 = pixelloop(
                        within2,
                        within5,
                        within10,
                        within20,
                        chargetotal,
                        labelbasic_numpy,
                        weights_numpy,
                        ADC_numpy,
                        ADCvalue_np,
                        overlay_numpy,
                        thresh_numpy,
                        h, h2)

            overlay_t = torch.from_numpy(overlay_numpy)
            thresh_t = torch.from_numpy(thresh_numpy)
            #calculate accuracies
            accuracy2 = (within2/chargetotal)*100
            accuracy5 = (within5/chargetotal)*100
            accuracy10 = (within10/chargetotal)*100
            accuracy20 = (within20/chargetotal)*100
            averageacc2 += accuracy2
            averageacc5 += accuracy5
            averageacc10 += accuracy10
            averageacc20 += accuracy20

            # print "Accuracy < 2: ", accuracy2
            # print "Accuracy < 5: ", accuracy5
            # print "Accuracy < 10: ", accuracy10
            # print "Accuracy < 20: ", accuracy20
            # print "  "

            # save created outputs
            if SAVE_DIFF:
                diff_out = larcv.as_image2d_meta(acc_t[ib,0,:,:], outmeta)
                ev_out_diff.append(diff_out)
            if SAVE_THRESHOLDS:
                thresh_out = larcv.as_image2d_meta(thresh_t[ib,0,:,:], outmeta)
                ev_out_thresh.append(thresh_out)
            if SAVE_OVERLAY:
                overlay_out = larcv.as_image2d_meta(overlay_t[ib,0,:,:], outmeta)
                ev_out_overlay.append(overlay_out)


            outputdata.set_id( ev_meta.run(), ev_meta.subrun(), ev_meta.event() )
            outputdata.save_entry()
            ientry += 1
            afterloop = time.time()

    print "-------------------------"
    print "Average Accuracies"
    print "<2: ", (averageacc2/nprocess_events)
    print "<5: ", (averageacc5/nprocess_events)
    print "<10: ", (averageacc10/nprocess_events)
    print "<20: ", (averageacc20/nprocess_events)
    print "-------------------------"
    # save 2d hist as png image
    c1 = TCanvas("diffs2D", "diffs2D", 600, 400)
    line = TLine(0,0,40,40)
    line.SetLineColor(632)
    h2.SetOption("COLZ")
    c1.SetLogz()
    h2.GetXaxis().SetTitle("True ADC value")
    h2.GetYaxis().SetTitle("Predicted ADC value")
    h2.Draw()
    line.Draw()
    c1.SaveAs(("diffs.png"))


    # stop input
    inputdata.stop()

    # save results
    outputdata.finalize()

    f.Write()
    f.Close()

    print "DONE."
    end = time.time()
    print "time to loop start: ", (beforeloop - start)
    print "time to do loop: ", (afterloop - beforeloop)
    print "time to finalize: ", (end - afterloop)
    print "total time: ", (end - start)
