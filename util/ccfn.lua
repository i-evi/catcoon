fn = {
  alias = {
    parameter = false,
    template  = "$output = $input;"
  },
  conv2d = {
    parameter = true,
    padding   = 0,
    offset    = 0,
    template  = "$output = cc_conv2d($input, "..
      "$pack.w, $pack.b, $stride, $padding, $offset, $name);"
  },
  batch_norm2d = {
    parameter = true,
    template  = "$output = cc_batch_norm2d($input, $pack.n, $name);"
  },
  fully_connected = {
    parameter = true,
    template  = "$output = cc_fully_connected($input, $pack.w, $pack.b, $name);"
  },
  add = {
    parameter = false,
    layers    = { "alpha" },
    template  = "$output = cc_elemwise($input, $alpha, '+', $name);"
  },
  avg_pool2d = {
    parameter = false,
    padding   = 0,
    offset    = 0,
    template  = "$output = cc_avg_pool2d($input, "..
      "$kernel, $stride, $padding, $offset, $name);"
  },
  max_pool2d = {
    parameter = false,
    padding   = 0,
    offset    = 0,
    template  = "$output = cc_max_pool2d($input, "..
      "$kernel, $stride, $padding, $offset, $name);"
  },
  relu = {
    parameter = false,
    name      = "#NULL", -- default to be `NULL`
    template  = "$output = cc_relu($input, $name);"
  },
  softmax = {
    parameter = false,
    name      = "#NULL", -- default to be `NULL`
    template  = "$output = cc_softmax($input, $name);"
  },
  reshape = {
    parameter = false,
    template = "$output = cc_reshape($input, $shape);"
  }
}