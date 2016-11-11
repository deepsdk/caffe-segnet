#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gflags/gflags.h>


#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;


DEFINE_string(model, "", "Model file(.prototxt) path.");
DEFINE_string(weights, "", "Weights file(.caffemodel) path.");
DEFINE_string(meanfile, "", "Mean file(.binaryproto) path.");
DEFINE_string(mean, "", "Mean BGR values seperated by commas.");
DEFINE_string(label, "", "Label file path.");
DEFINE_string(output, "", "Output mask image file path.");

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& mean,
             const string& label_file,
             const string& output_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

  void OutputMaskFile();

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  cv::Scalar mean_bgr_;
  std::vector<string> labels_;
  std::vector<cv::Vec3b> colors_;
  std::string output_file_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& mean,
                       const string& label_file,
                       const string& output_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->set_debug_info(true);
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  if (mean_file != ""){
    SetMean(mean_file);
  }else if (mean != ""){
    stringstream ss(mean);
    vector<int> bgr;
    string token;
    while(std::getline(ss, token, ',')) {
      int v = atoi(token.c_str());
      bgr.push_back(v);
    }
    CHECK(bgr.size() == 3)
      << "Mean BGR values should be 3 integers.";
    mean_bgr_ = cv::Scalar(bgr[0], bgr[1], bgr[2]);
  }

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line)){
//      labels_.push_back(string(line));

    std::stringstream ss(line);
    string name;
    std::getline(ss, name, ' ');
    labels_.push_back(name);

    string rgb;
    std::getline(ss, rgb, ' ');

    std::stringstream srgb(rgb);
    string r;
    string g;
    string b;
    std::getline(srgb, r, ',');
    std::getline(srgb, g, ',');
    std::getline(srgb, b, ',');

    cv::Vec3b bgr(atoi(b.c_str()), atoi(g.c_str()), atoi(r.c_str()));
    colors_.push_back(bgr);
  }

//    Blob<float>* output_layer = net_->output_blobs()[0];
//    CHECK_EQ(labels_.size(), output_layer->channels())
//      << "Number of labels is different from the output layer dimension.";

  output_file_ = output_file;
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

//  N = std::min<int>(labels_.size(), N);
//  std::vector<int> maxN = Argmax(output, N);
//  std::vector<Prediction> predictions;
//  for (int i = 0; i < N; ++i) {
//    int idx = maxN[i];
//    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
//  }
//  return predictions;
  std::vector<Prediction> predictions;
  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

void Classifier::OutputMaskFile() {
  Blob<float>* blob = net_->output_blobs()[0];
  std::cout << "dbg>output_layer=" << blob->shape_string() << std::endl;

  const int channels = blob->channels();
  const int height = blob->height();
  const int width = blob->width();
  const float* data = blob->cpu_data();

  for (int c = 0; c < channels; ++c) {
    cv::Mat img = cv::Mat::zeros(height, width, CV_8UC3);
    for (int h = 0; h < height; ++h) {
      cv::Vec3b* ptr = img.ptr<cv::Vec3b>(h);
      for (int w = 0; w < width; ++w) {
        int id = c * height * width + h * width + w;
        int cls = data[id];
        ptr[w] = colors_[cls];
      }
    }
    cv::imwrite(output_file_, img);
    break;
  }
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->ForwardPrefilled();

  OutputMaskFile();

  /* Copy the output layer to a std::vector */
//  const float* begin = output_layer->cpu_data();
//  const float* end = begin + output_layer->channels();
//  return std::vector<float>(begin, end);
  return std::vector<float>();
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_){
    cv::resize(sample, sample_resized, input_geometry_);
  } else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  if (!mean_.empty()){
    cv::subtract(sample_float, mean_, sample_normalized);
  }else{
//    std::vector<cv::Mat> planes;
//    cv::split(sample_float, planes);
//    for(int i = 0; i < planes.size(); i++){
//      stringstream ss;
//      ss << "origin" << i << ".png";
//      cv::imwrite(ss.str(), planes[i]);
//    }

    sample_normalized = sample_float - mean_bgr_;
  }

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);

  string model_file   = FLAGS_model;
  string trained_file = FLAGS_weights;
  string mean_file    = FLAGS_meanfile;
  string mean    = FLAGS_mean;
  string label_file   = FLAGS_label;
  string output_file   = FLAGS_output;
  Classifier classifier(model_file, trained_file, mean_file, mean, label_file, output_file);

  string file = argv[1];

  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  std::vector<Prediction> predictions = classifier.Classify(img);

//  /* Print the top N predictions. */
//  for (size_t i = 0; i < predictions.size(); ++i) {
//    Prediction p = predictions[i];
//    std::cout << p.first << "\t"
//       << std::fixed << std::setprecision(4) << p.second << std::endl;
//  }
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
