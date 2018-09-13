// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>





#include "caffe/fast_rcnn_layers.hpp"

using std::max;
using std::min;

namespace caffe {

	template <typename Dtype>
	inline __device__ Dtype gpu_atomic_add(const Dtype val, Dtype* address);

	template <>
	inline __device__
		double gpu_atomic_add(const double val, double* address) {
		return atomicAdd(address, val);
	}
	template <>
	inline __device__
		float gpu_atomic_add(const float val, float* address) {
		return atomicAdd(address, val);
	}

	template <typename Dtype>
	__device__ Dtype bilinear_interpolate(const Dtype* bottom_data,
		const int height, const int width,
		Dtype y, Dtype x,
		const int index /* index for debug only*/) {

		// deal with cases that inverse elements are out of feature map boundary
		if (y < -1.0 || y > height || x < -1.0 || x > width) {
			//empty
			return 0;
		}

		if (y <= 0) y = 0;
		if (x <= 0) x = 0;

		int y_low = (int)y;
		int x_low = (int)x;
		int y_high;
		int x_high;

		if (y_low >= height - 1) {
			y_high = y_low = height - 1;
			y = (Dtype)y_low;
		}
		else {
			y_high = y_low + 1;
		}

		if (x_low >= width - 1) {
			x_high = x_low = width - 1;
			x = (Dtype)x_low;
		}
		else {
			x_high = x_low + 1;
		}

		Dtype ly = y - y_low;
		Dtype lx = x - x_low;
		Dtype hy = 1. - ly, hx = 1. - lx;
		// do bilinear interpolation
		Dtype v1 = bottom_data[y_low * width + x_low];
		Dtype v2 = bottom_data[y_low * width + x_high];
		Dtype v3 = bottom_data[y_high * width + x_low];
		Dtype v4 = bottom_data[y_high * width + x_high];
		Dtype w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

		Dtype val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

		return val;
	}

	template <typename Dtype>
	__global__ void ROIAlignForward(const int nthreads, const Dtype* bottom_data,
		const Dtype spatial_scale, const int channels, const int height,
		const int width, const int pooled_height, const int pooled_width,const int sampling_ratio,
		const Dtype* bottom_rois, Dtype* top_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			// (n, c, ph, pw) is an element in the pooled output
			int pw = index % pooled_width;
			int ph = (index / pooled_width) % pooled_height;
			int c = (index / pooled_width / pooled_height) % channels;
			int n = index / pooled_width / pooled_height / channels;
			
			const Dtype* offset_bottom_rois = bottom_rois + n * 5;
			//bottom_rois += n * 5;
			int roi_batch_ind = offset_bottom_rois[0];
			Dtype roi_start_w = offset_bottom_rois[1] * spatial_scale;
			Dtype roi_start_h = offset_bottom_rois[2] * spatial_scale;
			Dtype roi_end_w = offset_bottom_rois[3] * spatial_scale;
			Dtype roi_end_h = offset_bottom_rois[4] * spatial_scale;

			// Force malformed ROIs to be 1x1
			Dtype roi_width = max(roi_end_w - roi_start_w + 1, 1.0);
			Dtype roi_height = max(roi_end_h - roi_start_h + 1, 1.0);
			Dtype bin_size_h = static_cast<Dtype>(roi_height)
				/ static_cast<Dtype>(pooled_height);
			Dtype bin_size_w = static_cast<Dtype>(roi_width)
				/ static_cast<Dtype>(pooled_width);

			const Dtype* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

			// We use roi_bin_grid to sample the grid and mimic integral
			int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
			int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

			// We do average (integral) pooling inside a bin
			const Dtype count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

			Dtype output_val = 0.;
			for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
			{
				const Dtype y = roi_start_h + ph * bin_size_h + static_cast<Dtype>(iy + .5f) * bin_size_h / static_cast<Dtype>(roi_bin_grid_h); // e.g., 0.5, 1.5
				for (int ix = 0; ix < roi_bin_grid_w; ix++)
				{
					const Dtype x = roi_start_w + pw * bin_size_w + static_cast<Dtype>(ix + .5f) * bin_size_w / static_cast<Dtype>(roi_bin_grid_w);

					Dtype val = bilinear_interpolate(offset_bottom_data, height, width, y, x, index);
					output_val += val;
				}
			}
			output_val /= count;

			top_data[index] = output_val;
		}
	}

	template <typename Dtype>
	void ROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* bottom_rois = bottom[1]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		//int* argmax_data = max_idx_.mutable_gpu_data();
		assert(sampling_ratio_ >= 0);
		
		int count = top[0]->count();
		// NOLINT_NEXT_LINE(whitespace/operators)
		ROIAlignForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, bottom_data, spatial_scale_, channels_, height_, width_,
			pooled_height_, pooled_width_, sampling_ratio_, bottom_rois, top_data);
		CUDA_POST_KERNEL_CHECK;
	}
	template <typename Dtype>
	__device__ void bilinear_interpolate_gradient(
		const int height, const int width,
		Dtype y, Dtype x,
		Dtype & w1, Dtype & w2, Dtype & w3, Dtype & w4,
		int & x_low, int & x_high, int & y_low, int & y_high,
		const int index /* index for debug only*/) {

		// deal with cases that inverse elements are out of feature map boundary
		if (y < -1.0 || y > height || x < -1.0 || x > width) {
			//empty
			w1 = w2 = w3 = w4 = 0.;
			x_low = x_high = y_low = y_high = -1;
			return;
		}

		if (y <= 0) y = 0;
		if (x <= 0) x = 0;

		y_low = (int)y;
		x_low = (int)x;

		if (y_low >= height - 1) {
			y_high = y_low = height - 1;
			y = (Dtype)y_low;
		}
		else {
			y_high = y_low + 1;
		}

		if (x_low >= width - 1) {
			x_high = x_low = width - 1;
			x = (Dtype)x_low;
		}
		else {
			x_high = x_low + 1;
		}

		Dtype ly = y - y_low;
		Dtype lx = x - x_low;
		Dtype hy = 1. - ly, hx = 1. - lx;

		// reference in forward
		// T v1 = bottom_data[y_low * width + x_low];
		// T v2 = bottom_data[y_low * width + x_high];
		// T v3 = bottom_data[y_high * width + x_low];
		// T v4 = bottom_data[y_high * width + x_high];
		// T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

		w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

		return;
	}
	template <typename Dtype>
	__global__ void ROIAlignBackward(const int nthreads, const Dtype* top_diff,
		const int num_rois, const Dtype spatial_scale,
		const int channels, const int height, const int width,
		const int pooled_height, const int pooled_width,const int sampling_ratio, Dtype* bottom_diff,
		const Dtype* bottom_rois) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			// (n, c, h, w) coords in pooled output
			int pw = index % pooled_width;
			int ph = (index / pooled_width) % pooled_height;
			int c = (index / pooled_width / pooled_height) % channels;
			int n = index / pooled_width / pooled_height / channels;

			const Dtype* offset_bottom_rois = bottom_rois + n * 5;
			int roi_batch_ind = offset_bottom_rois[0];

			// Do not using rounding; this implementation detail is critical
			Dtype roi_start_w = offset_bottom_rois[1] * spatial_scale;
			Dtype roi_start_h = offset_bottom_rois[2] * spatial_scale;
			Dtype roi_end_w = offset_bottom_rois[3] * spatial_scale;
			Dtype roi_end_h = offset_bottom_rois[4] * spatial_scale;
			// T roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
			// T roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
			// T roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
			// T roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

			// Force malformed ROIs to be 1x1
			Dtype roi_width = max(roi_end_w - roi_start_w, (Dtype)1.);
			Dtype roi_height = max(roi_end_h - roi_start_h, (Dtype)1.);
			Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
			Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);

			Dtype* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;

			int top_offset = (n * channels + c) * pooled_height * pooled_width;
			const Dtype* offset_top_diff = top_diff + top_offset;
			const Dtype top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

			// We use roi_bin_grid to sample the grid and mimic integral
			int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
			int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

			// We do average (integral) pooling inside a bin
			const Dtype count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

			for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
			{
				const Dtype y = roi_start_h + ph * bin_size_h + static_cast<Dtype>(iy + .5f) * bin_size_h / static_cast<Dtype>(roi_bin_grid_h); // e.g., 0.5, 1.5
				for (int ix = 0; ix < roi_bin_grid_w; ix++)
				{
					const Dtype x = roi_start_w + pw * bin_size_w + static_cast<Dtype>(ix + .5f) * bin_size_w / static_cast<Dtype>(roi_bin_grid_w);

					Dtype w1, w2, w3, w4;
					int x_low, x_high, y_low, y_high;

					bilinear_interpolate_gradient(height, width, y, x,
						w1, w2, w3, w4,
						x_low, x_high, y_low, y_high,
						index);

					Dtype g1 = top_diff_this_bin * w1 / count;
					Dtype g2 = top_diff_this_bin * w2 / count;
					Dtype g3 = top_diff_this_bin * w3 / count;
					Dtype g4 = top_diff_this_bin * w4 / count;

					if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0)
					{
						
						gpu_atomic_add(static_cast<Dtype>(g1), offset_bottom_diff + y_low * width + x_low);
						gpu_atomic_add(static_cast<Dtype>(g2), offset_bottom_diff + y_low * width + x_high);
						gpu_atomic_add(static_cast<Dtype>(g3), offset_bottom_diff + y_high * width + x_low);
						gpu_atomic_add(static_cast<Dtype>(g4), offset_bottom_diff + y_high * width + x_high);
						
						/*
						atomicAdd(offset_bottom_diff + y_low * width + x_low, static_cast<Dtype>(g1));
						atomicAdd(offset_bottom_diff + y_low * width + x_high, static_cast<Dtype>(g2));
						atomicAdd(offset_bottom_diff + y_high * width + x_low, static_cast<Dtype>(g3));
						atomicAdd(offset_bottom_diff + y_high * width + x_high, static_cast<Dtype>(g4));
						*/
					} // if
				} // ix
			} // iy
		} // CUDA_1D_KERNEL_LOOP
	}

	template <typename Dtype>
	void ROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (!propagate_down[0]) {
			return;
		}
		const Dtype* bottom_rois = bottom[1]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const int count = top[0]->count();
		caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom_diff);
		//const int* argmax_data = max_idx_.gpu_data();
		// NOLINT_NEXT_LINE(whitespace/operators)
		ROIAlignBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, top_diff,  bottom[1]->num(), spatial_scale_, channels_,
			height_, width_, pooled_height_, pooled_width_,sampling_ratio_, bottom_diff, bottom_rois);
		CUDA_POST_KERNEL_CHECK;
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ROIAlignLayer);

}  // namespace caffe
