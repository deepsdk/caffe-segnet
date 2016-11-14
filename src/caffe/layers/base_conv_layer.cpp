#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  CHECK(!conv_param.has_kernel_size() !=
      !(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(conv_param.has_kernel_size() ||
      (conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!conv_param.has_pad() && conv_param.has_pad_h()
      && conv_param.has_pad_w())
      || (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!conv_param.has_stride() && conv_param.has_stride_h()
      && conv_param.has_stride_w())
      || (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = conv_param.kernel_size();
  } else {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!conv_param.has_pad_h()) {
    pad_h_ = pad_w_ = conv_param.pad();
  } else {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  }
  if (!conv_param.has_stride_h()) {
    stride_h_ = stride_w_ = conv_param.stride();
  } else {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = kernel_w_ == 1 && kernel_h_ == 1
      && stride_h_ == 1 && stride_w_ == 1 && pad_h_ == 0 && pad_w_ == 0;
  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(
        conv_out_channels_, conv_in_channels_ / group_, kernel_h_, kernel_w_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      vector<int> bias_shape(1, num_output_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  std::cout << "dbg>BaseConvolutionLayer>Reshape enter-------------" << std::endl;

  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " convolution kernel.";
  std::cout << "    num_=" << num_ << ", height_=" <<  height_ << ", width_=" << width_ << ", channels_=" << channels_ << std::endl;
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  // Shape the tops.
  compute_output_shape();
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
  std::cout << "    num_=" << num_ << ", num_output_=" << num_output_ <<", height_out_=" <<  height_out_ << ", width_out_=" << width_out_ << std::endl;

  if (reverse_dimensions()) {
    conv_in_height_ = height_out_;
    conv_in_width_ = width_out_;
    conv_out_spatial_dim_ = height_ * width_;
  } else {
    conv_in_height_ = height_;
    conv_in_width_ = width_;
    conv_out_spatial_dim_ = height_out_ * width_out_;
  }
  std::cout << "    conv_in_height_=" << conv_in_height_ << ", conv_in_width_=" << conv_in_width_ << ", conv_out_spatial_dim_=" << conv_out_spatial_dim_ << std::endl; 
  kernel_dim_ = conv_in_channels_ * kernel_h_ * kernel_w_;
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_ / group_;
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_ / group_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  std::cout << "    kernel_dim_=" << kernel_dim_ << ", weight_offset_=" << weight_offset_ << ", col_offset_=" << col_offset_ << ", output_offset_=" << output_offset_ << std::endl; 
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  if (reverse_dimensions()) {
    col_buffer_.Reshape(1, kernel_dim_, height_, width_);
  } else {
    col_buffer_.Reshape(1, kernel_dim_, height_out_, width_out_);
  }
  std::cout << "----------------------------------" << std::endl;
  std::cout << "    col_buffer_.shape=" << col_buffer_.shape_string() << std::endl; 
  std::cout << "    kernel_dim_=" << kernel_dim_ << " <=  conv_in_channels_=" << conv_in_channels_ << " * kernel_h_=" << kernel_h_ << " * kernel_w_=" << kernel_w_ << std::endl; 

  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, height_out_ * width_out_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  std::cout << "dbg>BaseConvolutionLayer>forward_cpu_gemm enter---------------" << std::endl;
  const Dtype* col_buff = input;
  Blob<Dtype> col_buffer_split;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
//      conv_im2col_cpu_split(input+0*3*3, col_buffer_.mutable_cpu_data(), 3);
      std::cout << "  col_buffer_=" << col_buffer_.shape_string() << std::endl;
      const Dtype* data = col_buffer_.cpu_data();
      for(int c = 0; c < 12; c++){
        for(int i = 0; i < 4; i++){
          int id = c*4 + i;
          std::cout << " " << data[id];
        }
        std::cout << std::endl;
        if ((c+1)%4 == 0){
          std::cout << std::endl;
        }
      }
    }
    col_buff = col_buffer_.cpu_data();
  }
  for(int n = 0; n < 2; n++){
    std::cout << "----------weights[" << n << "]" << std::endl;
    for(int c = 0; c < 3; c++){
      for(int i = 0; i < 4; i++){
        int id = n * 3 * 4 + c*4 + i;
        std::cout << " " << weights[id];
      }
      std::cout << std::endl;
      if ((c+1)%4 == 0){
        std::cout << std::endl;
      }
    }
  }

  //////////////////////////////////////////////////////

  Blob<Dtype> weight_split;
  int s[] = {2,1,2,2};
  vector<int> shape(s, s+4);
  weight_split.Reshape(shape);

  Blob<Dtype> output_split;
  int ss[] = {1,2,2,2};
  vector<int> shape2(ss, ss+4);
  output_split.Reshape(shape2);

  {
    col_buffer_split.Reshape(col_buffer_.shape(0), (col_buffer_.shape(1)+1)/3, 
        col_buffer_.shape(2), col_buffer_.shape(3));
    conv_im2col_cpu_split(input+0*3*3, col_buffer_split.mutable_cpu_data(), 3);
    std::cout << "  col_buffer_split=" << col_buffer_split.shape_string() << std::endl;
    const Dtype* data2 = col_buffer_split.cpu_data();
    for(int c = 0; c < col_buffer_split.shape(1); c++){
      for(int i = 0; i < col_buffer_split.shape(2) * col_buffer_split.shape(3); i++){
        int id = c*4 + i;
        std::cout << " " << data2[id];
      }
      std::cout << std::endl;
      if ((c+1)%4 == 0){
        std::cout << std::endl;
      }
    }

    std::cout << "------w.shape=" << weight_split.shape_string() << std::endl;
    Dtype* w = weight_split.mutable_cpu_data();
    int idTo = 0;
    for(int n = 0; n < 2; n++){
      for(int c = 0; c < 3; c++){
        if (c == 0){
          for(int i = 0; i < 2*2; i++){
            int idFrom = n * 3 * 4 + c*4 + i;
            w[idTo] = weights[idFrom];
            idTo++;
          }
        }
      }
    }
    for(int n = 0; n < 2; n++){
      std::cout << "  w[" << n << "]" << std::endl;
      for(int c = 0; c < 1; c++){
        for(int i = 0; i < 2*2; i++){
          int id = n * 1 * 4 + c*4 + i;
          std::cout << " " << w[id];
        }
        std::cout << std::endl;
        if ((c+1)%4 == 0){
          std::cout << std::endl;
        }
      }
    }


    int g = 0;
    const Dtype* col = col_buffer_split.cpu_data();
    Dtype* out = output_split.mutable_cpu_data();
    int kernel_dim = (conv_in_channels_+1)/3 * kernel_h_ * kernel_w_;

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim / group_,
        (Dtype)1., w, col,
        (Dtype)0., out);

    for(int n = 0; n < output_split.shape(0); n++){
      std::cout << " out[" << n << "]" << std::endl;
      for(int c = 0; c < output_split.shape(1); c++){
        for(int i = 0; i < output_split.shape(2)*output_split.shape(3); i++){
          int id = n * output_split.shape(1) * output_split.shape(2)*output_split.shape(3) + 
            c*output_split.shape(2)*output_split.shape(3) + i;
          std::cout << " " << out[id];
        }
        std::cout << std::endl;
        if ((c+1)%4 == 0){
          std::cout << std::endl;
        }
      }
    }
  }

  {
    col_buffer_split.Reshape(col_buffer_.shape(0), (col_buffer_.shape(1)+1)/3, 
        col_buffer_.shape(2), col_buffer_.shape(3));
    conv_im2col_cpu_split(input+1*3*3, col_buffer_split.mutable_cpu_data(), 3);
    std::cout << "  col_buffer_split=" << col_buffer_split.shape_string() << std::endl;
    const Dtype* data2 = col_buffer_split.cpu_data();
    for(int c = 0; c < col_buffer_split.shape(1); c++){
      for(int i = 0; i < col_buffer_split.shape(2) * col_buffer_split.shape(3); i++){
        int id = c*4 + i;
        std::cout << " " << data2[id];
      }
      std::cout << std::endl;
      if ((c+1)%4 == 0){
        std::cout << std::endl;
      }
    }


    std::cout << "------w.shape=" << weight_split.shape_string() << std::endl;
    Dtype* w = weight_split.mutable_cpu_data();
    int idTo = 0;
    for(int n = 0; n < 2; n++){
      for(int c = 0; c < 3; c++){
        if (c == 1){
          for(int i = 0; i < 2*2; i++){
            int idFrom = n * 3 * 4 + c*4 + i;
            w[idTo] = weights[idFrom];
            idTo++;
          }
        }
      }
    }
    for(int n = 0; n < 2; n++){
      std::cout << "  w[" << n << "]" << std::endl;
      for(int c = 0; c < 1; c++){
        for(int i = 0; i < 2*2; i++){
          int id = n * 1 * 4 + c*4 + i;
          std::cout << " " << w[id];
        }
        std::cout << std::endl;
        if ((c+1)%4 == 0){
          std::cout << std::endl;
        }
      }
    }


    int g = 0;
    const Dtype* col = col_buffer_split.cpu_data();
    Dtype* out = output_split.mutable_cpu_data();
    int kernel_dim = (conv_in_channels_+1)/3 * kernel_h_ * kernel_w_;

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim / group_,
        (Dtype)1., w, col,
        (Dtype)0., out);

    for(int n = 0; n < output_split.shape(0); n++){
      std::cout << " out[" << n << "]" << std::endl;
      for(int c = 0; c < output_split.shape(1); c++){
        for(int i = 0; i < output_split.shape(2)*output_split.shape(3); i++){
          int id = n * output_split.shape(1) * output_split.shape(2)*output_split.shape(3) + 
            c*output_split.shape(2)*output_split.shape(3) + i;
          std::cout << " " << out[id];
        }
        std::cout << std::endl;
        if ((c+1)%4 == 0){
          std::cout << std::endl;
        }
      }
    }
  }

  {
    col_buffer_split.Reshape(col_buffer_.shape(0), (col_buffer_.shape(1)+1)/3, 
        col_buffer_.shape(2), col_buffer_.shape(3));
    conv_im2col_cpu_split(input+2*3*3, col_buffer_split.mutable_cpu_data(), 3);
    std::cout << "  col_buffer_split=" << col_buffer_split.shape_string() << std::endl;
    const Dtype* data2 = col_buffer_split.cpu_data();
    for(int c = 0; c < col_buffer_split.shape(1); c++){
      for(int i = 0; i < col_buffer_split.shape(2) * col_buffer_split.shape(3); i++){
        int id = c*4 + i;
        std::cout << " " << data2[id];
      }
      std::cout << std::endl;
      if ((c+1)%4 == 0){
        std::cout << std::endl;
      }
    }

    std::cout << "------w.shape=" << weight_split.shape_string() << std::endl;
    Dtype* w = weight_split.mutable_cpu_data();
    int idTo = 0;
    for(int n = 0; n < 2; n++){
      for(int c = 0; c < 3; c++){
        if (c == 2){
          for(int i = 0; i < 2*2; i++){
            int idFrom = n * 3 * 4 + c*4 + i;
            w[idTo] = weights[idFrom];
            idTo++;
          }
        }
      }
    }
    for(int n = 0; n < 2; n++){
      std::cout << "  w[" << n << "]" << std::endl;
      for(int c = 0; c < 1; c++){
        for(int i = 0; i < 2*2; i++){
          int id = n * 1 * 4 + c*4 + i;
          std::cout << " " << w[id];
        }
        std::cout << std::endl;
        if ((c+1)%4 == 0){
          std::cout << std::endl;
        }
      }
    }


    int g = 0;
    const Dtype* col = col_buffer_split.cpu_data();
    Dtype* out = output_split.mutable_cpu_data();
    int kernel_dim = (conv_in_channels_+1)/3 * kernel_h_ * kernel_w_;

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim / group_,
        (Dtype)1., w, col,
        (Dtype)0., out);

    for(int n = 0; n < output_split.shape(0); n++){
      std::cout << " out[" << n << "]" << std::endl;
      for(int c = 0; c < output_split.shape(1); c++){
        for(int i = 0; i < output_split.shape(2)*output_split.shape(3); i++){
          int id = n * output_split.shape(1) * output_split.shape(2)*output_split.shape(3) + 
            c*output_split.shape(2)*output_split.shape(3) + i;
          std::cout << " " << out[id];
        }
        std::cout << std::endl;
        if ((c+1)%4 == 0){
          std::cout << std::endl;
        }
      }
    }
  }  
  //////////////////////////////////////////////////////


  for (int g = 0; g < group_; ++g) {
    std::cout << "dbg>base>conv_out_channels_=" << conv_out_channels_ << std::endl;
    std::cout << "dbg>base>group_=" << group_ << std::endl;
    std::cout << "dbg>base>conv_out_spatial_dim_=" << conv_out_spatial_dim_ << std::endl;
    std::cout << "dbg>base>kernel_dim_=" << kernel_dim_ << std::endl;
    std::cout << "dbg>base>weight_offset_=" << weight_offset_ << std::endl;
    std::cout << "dbg>base>col_offset_=" << col_offset_ << std::endl;
    std::cout << "dbg>base>output_offset_=" << output_offset_ << std::endl;

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
