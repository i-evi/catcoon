fn = {
  conv2d = {
    parameter = true,
    padding   = 0,
    offset    = 0,
    template  = "$output = cc_conv2d($input, "..
      "$pack.w, $pack.b, $stride, $padding, $offset, $name);"
  },
  fully_connected = {
    parameter = true,
    template  = "$output = cc_fully_connected($input, $pack.w, $pack.b, $name);"
  },
  add = {
    parameter = false,
    layers    = { "alpha" }, 
    template  = "$output = cc_add($input, $alpha, $name);"
  },
  max_pool2d = {
    parameter = false,
    template  = "$output = cc_max_pool2d($input, $stride, $name);"
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