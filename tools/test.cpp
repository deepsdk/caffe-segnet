#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <gflags/gflags.h>

using namespace caffe;  // NOLINT(build/namespaces)

DEFINE_string(model, "", "Model file(.prototxt) path.");
DEFINE_string(weights, "", "Weights file(.caffemodel) path.");
DEFINE_string(meanfile, "", "Mean file(.binaryproto) path.");
DEFINE_string(mean, "", "Mean BGR values seperated by commas.");
DEFINE_string(label, "", "Label file path.");
DEFINE_string(output, "", "Output mask image file path.");

void dump_blob(shared_ptr<Blob<float> > blob){
  std::cout << "---------------" << blob->shape_string() << "------------------" << std::endl;
  const float* data = blob->cpu_data();
  for(int n = 0; n < blob->shape(0); n++){
    for(int c = 0; c < blob->shape(1); c++){
      std::cout << "n=" << n << ", c=" << c << std::endl;
      for(int h = 0; h < blob->shape(2); h++){
        for(int w = 0; w < blob->shape(3); w++){
          int i = n * blob->shape(1) * blob->shape(2) * blob->shape(3) + 
            c * blob->shape(2) * blob->shape(3) + h * blob->shape(3) + w;
          std::cout << "  " << data[i];
        }
        std::cout << std::endl;
      }
    }
  }
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);

  string model_file   = FLAGS_model;
  string trained_file = FLAGS_weights;
  string mean_file    = FLAGS_meanfile;
  string mean    = FLAGS_mean;
  string label_file   = FLAGS_label;

  Caffe::set_mode(Caffe::CPU);
  Net<float> net(model_file, TEST);

  if (trained_file == "full"){
    std::cout << "---------------------------------- image" << std::endl;
    float *data = net.blobs()[0]->mutable_cpu_data();
    float image[4*3*3] = {
      1,2,0,1,1,3,0,2,2,
      0,2,1,0,3,2,1,1,0,
      1,2,1,0,1,3,3,3,2,
      0,0,0,0,0,0,0,0,0,
    };
    for(int i = 0; i < 4*3*3; i++){
      data[i] = image[i];
    }
    dump_blob(net.blobs()[0]);

    std::cout << "---------------------------------- kernel" << std::endl;
    for(int i = 0; i < net.layers().size(); i++){
      vector<shared_ptr<Blob<float> > > blobs = net.layers()[i]->blobs();
      std::cout << net.layer_names()[i] << ": " << blobs[0]->shape_string() << std::endl;
      float kernel[2*4*2*2] = {
        1,1,2,2,
        1,1,1,1,
        0,1,1,0,
        1,1,1,1,

        1,0,0,1,
        2,1,2,1,
        1,2,2,0,
        1,1,1,1,
      };
      float* data = blobs[0]->mutable_cpu_data();
      for(int i = 0; i < 2*4*2*2; i++){
        data[i] = kernel[i];
      }
      dump_blob(blobs[0]);
    }
    
    net.ForwardPrefilled();

    std::cout << "---------------------------------- output full" << std::endl;
    dump_blob(net.blobs()[1]);

  }else if (trained_file == "half1"){
    std::cout << "---------------------------------- image half1" << std::endl;
    float *data = net.blobs()[0]->mutable_cpu_data();
    float image[2*3*3] = {
      1,2,0,1,1,3,0,2,2,
      0,2,1,0,3,2,1,1,0,
//      1,2,1,0,1,3,3,3,2,
    };
    for(int i = 0; i < 2*3*3; i++){
      data[i] = image[i];
    }
    dump_blob(net.blobs()[0]);

    std::cout << "---------------------------------- kernel" << std::endl;
    for(int i = 0; i < net.layers().size(); i++){
      vector<shared_ptr<Blob<float> > > blobs = net.layers()[i]->blobs();
      std::cout << net.layer_names()[i] << ": " << blobs[0]->shape_string() << std::endl;
      float kernel[2*2*2*2] = {
        1,1,2,2,
        1,1,1,1,
 //       0,1,1,0,

        1,0,0,1,
        2,1,2,1,
  //      1,2,2,0,
      };
      float* data = blobs[0]->mutable_cpu_data();
      for(int i = 0; i < 2*2*2*2; i++){
        data[i] = kernel[i];
      }
      dump_blob(blobs[0]);
    }

    net.ForwardPrefilled();

    std::cout << "---------------------------------- output half1" << std::endl;
    dump_blob(net.blobs()[1]);

  }else if (trained_file == "half2"){
    std::cout << "---------------------------------- image half2" << std::endl;
    float *data = net.blobs()[0]->mutable_cpu_data();
    float image[1*3*3] = {
//      1,2,0,1,1,3,0,2,2,
//      0,2,1,0,3,2,1,1,0,
      1,2,1,0,1,3,3,3,2,
    };
    for(int i = 0; i < 1*3*3; i++){
      data[i] = image[i];
    }
    dump_blob(net.blobs()[0]);

    std::cout << "---------------------------------- kernel" << std::endl;
    for(int i = 0; i < net.layers().size(); i++){
      vector<shared_ptr<Blob<float> > > blobs = net.layers()[i]->blobs();
      std::cout << net.layer_names()[i] << ": " << blobs[0]->shape_string() << std::endl;
      float kernel[2*1*2*2] = {
//        1,1,2,2,
//        1,1,1,1,
        0,1,1,0,

//        1,0,0,1,
//        2,1,2,1,
        1,2,2,0,
      };
      float* data = blobs[0]->mutable_cpu_data();
      for(int i = 0; i < 2*1*2*2; i++){
        data[i] = kernel[i];
      }
      dump_blob(blobs[0]);
    }

    net.ForwardPrefilled();

    std::cout << "---------------------------------- output half2" << std::endl;
    dump_blob(net.blobs()[1]);

  }else if (trained_file == "single"){
    std::cout << "---------------------------------- image" << std::endl;
    float *data = net.blobs()[0]->mutable_cpu_data();
    float image[3*3*3] = {
      1,2,0,1,1,3,0,2,2,
      0,2,1,0,3,2,1,1,0,
      1,2,1,0,1,3,3,3,2,
    };
    for(int i = 0; i < 1*3*3; i++){
      data[i] = image[i];
    }
    dump_blob(net.blobs()[0]);

    std::cout << "---------------------------------- kernel" << std::endl;
    for(int i = 0; i < net.layers().size(); i++){
      vector<shared_ptr<Blob<float> > > blobs = net.layers()[i]->blobs();
      std::cout << net.layer_names()[i] << ": " << blobs[0]->shape_string() << std::endl;
      float kernel[2*3*2*2] = {
        1,1,2,2,
        1,1,1,1,
        0,1,1,0,

        1,0,0,1,
        2,1,2,1,
        1,2,2,0,
      };
      float* data = blobs[0]->mutable_cpu_data();
      for(int i = 0; i < 1*1*2*2; i++){
        data[i] = kernel[i];
      }
      dump_blob(blobs[0]);
    }
    
    net.ForwardPrefilled();

    std::cout << "---------------------------------- output single" << std::endl;
    dump_blob(net.blobs()[1]);

  }

}
