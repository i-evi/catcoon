# Catcoon

![catcoon](catcoon.png?raw=ture)摸鱼玩饥荒时写的 Catcoon 是一个 C 语言实现的前向传播的神经网络框架。把你的网络移植到各种奇奇怪怪的平台上吧！

* 在正式的文档规范确定前，部分文档内容暂时写在这里(`readme.md`)。

* Catcoon 本体是使用 C 语言进行开发的，方便移植到各种平台，因此 bind 到脚本的话会首先选择 Lua，将在未来提供 Lua 支持。

* Catcoon 建议兼容 ISO C89 标准。考虑到性能，不必完全遵循 C89，但需要注明。

* 编码规范不做强制，建议参考 [Linux kernel coding style](https://www.kernel.org/doc/html/v4.10/process/coding-style.html)，现有的代码几乎遵守了此规范。

## 数据类型(Datatypes, cc_dtype)

`cc_dtype.h` 中定义 Catcoon 支持的数据类型，以便在各种平台之间移植。

```c
#define _DT_UINT8   unsigned char
...
#define CC_UINT8    0x03
...
typedef _DT_UINT8   cc_uint8;
...
```

## Tensor 管理器(Tensor Manager, cc_tsrmgr)

Catcoon 使用 tensor 进行计算。神经网络的计算过程中，经常产生新的 tensor。使用 tensor 管理器能够自动管理所有 tensor。`cc_tsrmgr.h` 声明了 tensor 管理器的基本的操作。

编译时定义 `AUTO_TSRMGR` 打开 tensor 管理器，取消这个预定义，即可以禁用自动 tensor 管理，参考 `makefile`。通常，加载一个模型之后，网络可能会持续处理多个样本，因此内存管理器使很多中间过程中产生的 tensor 驻留在内存中，这样做能够减少系统内存管理开销并且提高性能。但是在某些资源极端匮乏的平台上可能会显得有些奢侈，如果有需要，可以在编译时禁用 tensor 管理器，手动管理内存。

在启动了自动 tensor 管理的情况下任何创建新 tensor 的操作都会自动向 `cc_tsrmgr` 注册新创建的 tensor。但是 `name` 是 `NULL` 的 tensor 不会被自动注册。

## 张量(Tensor, cc_tensor)

`cc_tensor.h` 定义了 `cc_tensor_t` 和一些基本操作。

```c
typedef struct {
    list_t *container;
	const char     *name;
	unsigned char  *data;
	const cc_int32 *shape;
	const cc_int32 *dtype;
} cc_tensor_t;
```

为了方便数据管理，tensor 实际存储在线性表 `container`。为了方便访问 tensor，提供了 `name`, `data`, `shape` 和 `dtype` 指针。

* `name`: tensor 的名字，内存管理器 `cc_tsrmgr` 是基于 tensor 的命名管理 tensor 的，因此在使用内存管理器的场合，tensor 的名字应该是唯一的。

* `data`: 指向 tensor 的实际数据。

* `shape`: tensor 的形状，这是一个数组，以 `0` 结束。

* `dtype`: 数据类型，参考 `cc_dtype.h`。

### 创建/释放 Tensor (cc_create_tensor/cc_free_tensor)

使用 `cc_create_tensor`/`cc_free_tensor` 创建/释放一个 tensor：

```c
cc_tensor_t *cc_create_tensor(cc_int32 *shape, cc_int32 dtype, const char *name);

void cc_free_tensor(cc_tensor_t *tensor);
```
一个简单的例子(`demo/simple.c`)：

```c
...
cc_tensor_t *tensor;
cc_int32 shape[] = {3, 3, 3, 0};
tensor = cc_create_tensor(shape, CC_FLOAT32, "tensor0");
cc_print_tensor_property(tensor);
cc_free_tensor(tensor);
...
```
`cc_print_tensor_property` 输出 tensor 的基本属性：

```bash
[00000003]: tensor: "tensor0", dtype: "cc_float32", shape: [3, 3, 3]
```
如果没有使用 `cc_tsrmgr`，需要调用 `cc_free_tensor` 手动释放资源。特别注意， `cc_free_tensor` 是手动管理内存的方式，如果使用 `cc_tsrmgr`，不推荐使用 `cc_free_tensor`，在 `cc_tsrmgr` 启动的情况下，几乎所有过程中创建的 tensor 都会被纳入 `cc_tsrmgr`，如非必要，不应该手动释放 tensor，可能造成空悬指针。

### 保存/加载 Tensor (cc_save_tensor/cc_load_tensor)

你可以把一个 tensor 保存到文件中，也可以通过一个文件恢复一个 tensor。在保存 tensor 时，你可以使用任意的文件名，tensor 的名字也被记录到文件中。加载 tensor 时，如果启用了自动 tensor 管理器，而且 tensor 管理器中已经存在了和加载的 tensor 同名的 tensor，已经在 tensor 管理器中的 tensor 会被加载的 tensor 覆盖。

```c
cc_tensor_t *cc_load_tensor(const char *filename);

void cc_save_tensor(cc_tensor_t *tensor, const char *filename);
```

在保存和加载 tensor 时，都不需要指定 tensor 的形状和名字，`cc_load_bin` 是一种从文件创建 tensor 特殊方法，在某些特殊场合使用：

```c
cc_tensor_t *cc_load_bin(const char *filename,
	const cc_int32 *shape, cc_dtype dtype, const char *name);
```

`cc_load_bin` 可以从二进制流文件中读取数据并且加载成新的 tensor，但是你需要为 tensor 指定形状和名字。

## 图像接口(Image, cc_image)

`cc_image` 提供了 tensor 和图像之间的转换功能。

```c
cc_tensor_t *cc_image2tensor(utim_image_t *img, const char *name);

utim_image_t *cc_tensor2image(cc_tensor_t *tensor);
```

图像操作的支持在 `util_image.h` 中定义。包含有一些常用功能：

**读取/保存图像**支持 `bmp`, `jpg`, `png`, `tga` 格式的图像文件。 其中 `jpg`, `png`, `tga` 文件的支持是通过 [stb](https://github.com/nothings/stb) 实现的，我不确定 [stb](https://github.com/nothings/stb) 在某些编译器或者硬件平台的兼容性如何，可以在 `makefile` 中禁用 [stb](https://github.com/nothings/stb)，如果 [stb](https://github.com/nothings/stb) 被禁用，只支持 `bmp` 图像文件的读写。

```c
utim_image_t *utim_read(const char *filename);

int utim_write(const char *filename, utim_image_t *img);

int utim_write_ctrl(const char *filename, utim_image_t *img, int comp, int quality);
```

**基本的图像预处理**包括灰度，缩放，简单的 2D 图形功能，参考 `util_image.h`。

## 全局功能配置(Global Function Config, global_fn_cfg)

可以把 Catcoon 移植到任何可能运行你的网络的平台，并且发挥平台的计算优势。作为范例，Catcoon 只提供基于一般 CPU(常见的 x86, ARM) 的算法。在 Catcoon 提供的 API 中一般不体现计算本身的实现(但是有例外，如内存操作密集的算法推荐使用 CPU 实现)。

例如 2D CNN 中常用的 Spatial Convolution，`cc_conv2d` 中提供的 API 如下:

```c
cc_tensor_t *cc_conv2d(cc_tensor_t *inp, cc_tensor_t *kernel,
	cc_tensor_t *bias, cc_int32 s, cc_int32 p, cc_int32 off, const char *name);
```

2D 卷积是逐通道进行的，在 `src/cc_conv2d.c` 用于实现某层 feature map 卷积的函数指针声明如下:

```c
extern void (*_conv2d)(void *inp, void *oup, cc_int32 x, cc_int32 y, cc_int32 oup_x,
		cc_int32 oup_y, cc_int32 sx, cc_int32 sy, void *filter, cc_int32 fw, cc_dtype dt);
```

函数指针 `_conv2d` 的值在全局功能配置 `global_fn_cfg.h` 中设置:

```c
void (*_conv2d)(void *inp, void *oup, cc_int32 x, cc_int32 y,
cc_int32 oup_x, cc_int32 oup_y, cc_int32 sx, cc_int32 sy, void *filter, cc_int32 fw, cc_dtype dt) = cc_cpu_conv2d;
```

这样，实际上卷积计算是 `cc_cpu_conv2d` 完成的。如果你想用其他版本的卷积实现(例如 GPU/FPGA)，在 `global_fn_cfg.h` 把 `_conv2d` 指向你的实现就行了。

## 实现一个完整的 CNN 网络

在 Catcoon 开发过程中，实现了一些通用的功能后，应该给出一些基于这些功能的具有代表性的网络作为参考。

`demo/lenet.c` 是一个修改于 LeNet 的网络，与 LeNet 有共同的技术特征，包括卷积，池化，全连接，Relu、Softmax 激活函数。*训练和运行参考[https://github.com/i-evi/catcoon-pytorch-model/lenet](https://github.com/i-evi/catcoon-pytorch-model/lenet)

*Catcoon 是前向传播的，所以模型的训练通过其他深度学习框架实现。Catcoon 不负责训练网络，它只是提供一个轻量级的框架，让你的网络能够快速简单的部署在各种平台上，并为一些特殊的专用计算平台移植提供方便。
