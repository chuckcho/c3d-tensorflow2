import caffe
import numpy as np
import os


DIR_MODEL = 'models'
pth_in = os.path.join(DIR_MODEL, 'train01_16_128_171_mean.binaryproto')
pth_out = os.path.join(DIR_MODEL, 'train01_16_128_171_mean.npy')


if __name__ == '__main__':
    # read raw binary data
    blob = caffe.proto.caffe_pb2.BlobProto()
    raw = open(pth_in, 'rb').read()

    # parse raw binary data
    blob.ParseFromString(raw)
    print 'Channels:%d  Length:%d  Height:%d  Width:%d' % \
            (blob.channels, blob.length, blob.height, blob.width)

    # convert to array and reshape
    #arr = np.array( caffe.io.blobproto_to_array(blob) )
    arr = np.asarray(blob.data)
    means = arr.reshape(blob.channels, blob.length, blob.height, blob.width)
    np.save(pth_out, means)
