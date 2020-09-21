#ifndef _CC_DSC2D_H_
#define _CC_DSC2D_H_

#ifdef __cplusplus
	extern "C" {
#endif

/* Depth-wise convolution 2d */
cc_tensor_t *cc_dw_conv2d(cc_tensor_t *inp,
	cc_tensor_t *kernel, cc_tensor_t *bias, cc_int32 s,
	cc_int32 p, cc_int32 off, const char *name);

/* Point-wise convolution 2d */
cc_tensor_t *cc_pw_conv2d(cc_tensor_t *inp,
	cc_tensor_t *kernel, cc_tensor_t *bias, const char *name);

#ifdef __cplusplus
	}
#endif

#endif /* _CC_DSC2D_H_ */
