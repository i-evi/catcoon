CC = gcc
AR = ar rc

# Build Ctrl Flag
BCTRL =
DIALECT = # -std=c89

INC   += -I ./src/
LINK  += $(LDFLAGS)
LINK  += -lm

# Debug flags, cc_assert
DEBUG += -DENABLE_CC_ASSERT
# AddressSanitizer for gcc/clang
DEBUG += # -g -fsanitize=address -fno-omit-frame-pointer

CFLAGS += -Wall # -Wpedantic

# Optimization
OPTIM += -Ofast -flto
# Enable OpenMP
OPTIM += -DENABLE_OPENMP -fopenmp

CFLAGS += $(DEBUG) $(WARN) $(OPTIM) $(DIALECT)

# Behavior configurations
# Enable automatic tensor manager
CFLAGS += -DAUTO_TSRMGR

# Configurations for utilities
UTIM_COND   =
UT_LIST_CFG = -DENABLE_FOPS

# Additional source/library configurations
# some experimental implementations of NN functions on cpu
ADDITIONAL += ecpufn

# 3rd party source/library configurations
# stb: Read/write jpg, png, tga format images.
# Will only support bmp image if disabled
3RDSRC_CFG += stb

# parg: Argv parser written in ANSI C
# Argv parser for apps
3RDSRC_CFG += parg

# seb: sequential encoded buffers
# Pack and save compressed models files
3RDSRC_CFG += seb

# zlib: General-purpose compression and decompression library
# Pack and save compressed models files
3RDSRC_CFG += zlib

# cJSON: Ultralightweight JSON parser in ANSI C.
3RDSRC_CFG += cjson

ifneq ($(findstring MINI, $(BCTRL)),)
  CC = tcc # tiny C compiler
  CFLAGS = -DAUTO_TSRMGR
  ADDITIONAL =
  3RDSRC_CFG = parg cjson
endif

ifneq ($(findstring $(DIALECT), -ansi -std=c89 -std=c90),)
  CFLAGS += -DCONFIG_STD_C89
  ifneq ($(findstring $(CC), gcc clang),)
  	WARN += -Wno-long-long
  endif
endif

ifneq ($(findstring ecpufn, $(ADDITIONAL)),)
  ALL_O  += ecpufn.o
  CFLAGS += -march=native -DENABLE_ECPUFN
  ifneq ($(findstring $(DIALECT), -ansi -std=c89 -std=c90),)
    $(error "sse2neon.h is NOT ansi/c89/c90 compatible") 
  endif
endif

ifneq ($(findstring stb, $(3RDSRC_CFG)),)
  UTIM_COND += -DUSE3RD_STB_IMAGE -I ./src/3rd_party/stb/
endif

ifneq ($(findstring parg, $(3RDSRC_CFG)),)
  ALL_O   += parg.o
  APP_INC += -I ./src/3rd_party/parg/
endif

ifneq ($(findstring seb, $(3RDSRC_CFG)),)
  ALL_O += seb.o fastlz.o
  UT_LIST_CFG += -DENABLE_SEB -I ./src/3rd_party/seb/
endif

ifneq ($(findstring zlib, $(3RDSRC_CFG)),)
  LINK        += # -lz
  UT_LIST_CFG += # -DENABLE_ZLIB
endif

ifneq ($(findstring cjson, $(3RDSRC_CFG)),)
  ALL_O   += cJSON.o
  APP_INC += -I ./src/3rd_party/cjson/
endif

OBJS_PATH = objs
LIBS_PATH = libs
APPS_PATH = apps
VPATH += $(OBJS_PATH)

ALL_O += \
catcoon.o cc_tensor.o cc_dtype.o cc_tsrmgr.o cc_fmap2d.o cc_pool2d.o \
cc_array.o cc_basic.o cc_actfn.o cc_fullycon.o cc_pad2d.o cc_cpufn.o \
cc_conv2d.o cc_normfn.o cc_image.o util_rbt.o util_list.o util_log.o \
util_vec.o util_image.o global_fn_cfg.o 

CATCOON_A = libcatcoon.a

APPS_DEMO = simple lenet vgg16
APPS_UTIL = lspkg mkpkg cclua

APP_NAMES  = $(APPS_DEMO) $(APPS_UTIL)
APP_INC   += $(INC)
APP_LINK  += $(LINK)

ifeq ($(OS), Windows_NT)
  RM    = del
  MV    = move
  RMDIR = rmdir /q/s
  MKDIR = mkdir
  APPS  = $(foreach v, $(APP_NAMES), $(APPS_PATH)/$(v).exe)
else
  RM    = rm -f
  MV    = mv
  RMDIR = rm -rf
  MKDIR = mkdir -p
  APPS  = $(foreach v, $(APP_NAMES), $(APPS_PATH)/$(v))
endif

$$: $(OBJS_PATH)/build $(LIBS_PATH)/build $(APPS_PATH)/build
	$(MAKE) all

$(OBJS_PATH)/build:
	$(MKDIR) $(OBJS_PATH) && echo objs > $@
$(LIBS_PATH)/build:
	$(MKDIR) $(LIBS_PATH) && echo libs > $@
$(APPS_PATH)/build:
	$(MKDIR) $(APPS_PATH) && echo apps > $@

all: $(APPS)

$(LIBS_PATH)/$(CATCOON_A): $(ALL_O)
	cd $(OBJS_PATH) && $(AR) ../$@ $(ALL_O)

%.o: ./src/%.c
	$(CC) -c -o $(OBJS_PATH)/$@ $< $(CFLAGS) $(INC)

# Apps For Linux
$(APPS_PATH)/%: ./demo/%.c $(LIBS_PATH)/$(CATCOON_A)
	$(CC) -o $@ $^ $(CFLAGS) $(APP_INC) $(APP_LINK)
$(APPS_PATH)/%: ./util/%.c $(LIBS_PATH)/$(CATCOON_A)
	$(CC) -o $@ $^ $(CFLAGS) $(APP_INC) $(APP_LINK)
# Apps For Windows
$(APPS_PATH)/%.exe: ./demo/%.c $(LIBS_PATH)/$(CATCOON_A)
	$(CC) -o $@ $^ $(CFLAGS) $(APP_INC) $(APP_LINK)
$(APPS_PATH)/%.exe: ./util/%.c $(LIBS_PATH)/$(CATCOON_A)
	$(CC) -o $@ $^ $(CFLAGS) $(APP_INC) $(APP_LINK)

global_fn_cfg.o : $(patsubst %, ./src/%, global_fn_cfg.h global_fn_cfg.c)

catcoon.o     : $(patsubst %, ./src/%, catcoon.h catcoon.c)
cc_actfn.o    : $(patsubst %, ./src/%, cc_actfn.h cc_actfn.c)
cc_array.o    : $(patsubst %, ./src/%, cc_array.h cc_array.c)
cc_basic.o    : $(patsubst %, ./src/%, cc_basic.h cc_basic.c)
cc_conv2d.o   : $(patsubst %, ./src/%, cc_conv2d.h cc_conv2d.c)
cc_dsc2d.o    : $(patsubst %, ./src/%, cc_dsc2d.h cc_dsc2d.c)
cc_cpufn.o    : $(patsubst %, ./src/%, cc_cpufn.h cc_cpufn.c)
cc_dtype.o    : $(patsubst %, ./src/%, cc_dtype.h cc_dtype.c)
cc_fmap2d.o   : $(patsubst %, ./src/%, cc_fmap2d.h cc_fmap2d.c)
cc_fullycon.o : $(patsubst %, ./src/%, cc_fullycon.h cc_fullycon.c)
cc_image.o    : $(patsubst %, ./src/%, cc_image.h cc_image.c)
cc_normfn.o   : $(patsubst %, ./src/%, cc_normfn.h cc_normfn.c)
cc_pad2d.o    : $(patsubst %, ./src/%, cc_pad2d.h cc_pad2d.c)
cc_pool2d.o   : $(patsubst %, ./src/%, cc_pool2d.h cc_pool2d.c)
cc_tsrmgr.o   : $(patsubst %, ./src/%, cc_tsrmgr.h cc_tsrmgr.c)
cc_tensor.o   : $(patsubst %, ./src/%, cc_tensor.h cc_tensor.c)

util_log.o   : $(patsubst %, ./src/%, util_log.h util_log.c)
util_vec.o   : $(patsubst %, ./src/%, util_vec.h util_vec.c)
util_rbt.o   : $(patsubst %, ./src/%, util_rbt.h util_rbt.c)
util_list.o  : $(patsubst %, ./src/%, util_list.h util_list.c)
	$(CC) -c -o $(OBJS_PATH)/$@ $(filter %.c, $^) $(CFLAGS) $(UT_LIST_CFG)
util_image.o : $(patsubst %, ./src/%, util_image.h util_image.c)
	$(CC) -c -o $(OBJS_PATH)/$@ $(filter %.c, $^) $(CFLAGS) $(UTIM_COND)

# Additional objs
ecpufn.o: ./src/additional/ecpufn/ecpufn*
	$(CC) -c -o $(OBJS_PATH)/$@ $(filter %.c, $^) $(CFLAGS)

# 3rd party objs
parg.o: ./src/3rd_party/parg/parg*
	$(CC) -c -o $(OBJS_PATH)/$@ $(filter %.c, $^) $(CFLAGS)

seb.o : ./src/3rd_party/seb/seb*
	$(CC) -c -o $(OBJS_PATH)/$@ $(filter %.c, $^) $(CFLAGS)
fastlz.o : ./src/3rd_party/seb/fastlz*
	$(CC) -c -o $(OBJS_PATH)/$@ $(filter %.c, $^) $(CFLAGS)

cJSON.o : ./src/3rd_party/cjson/cJSON*
	$(CC) -c -o $(OBJS_PATH)/$@ $(filter %.c, $^) $(CFLAGS)

minimal:
	$(MAKE) "BCTRL = MINI"

clean:
	$(RMDIR) $(OBJS_PATH) $(LIBS_PATH) $(APPS_PATH)
