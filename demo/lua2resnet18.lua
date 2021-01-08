loadfile("./util/lua2cc.lua")()
loadfile("./util/ccfn.lua")()

resnet18 = new_context({
  name = "resnet18",
  inputs = {
    "in"
  },
  outputs = {
    "out"
  }
})

features   = new_module()
classifier = new_module()

make_basic_block = function(module, serial, stride)
  module:append(fn.conv2d      , { stride = stride, padding = 1 })
  module:append(fn.batch_norm2d)
  module:append(fn.relu        , { name = "#NULL" })
  module:append(fn.conv2d      , { stride = 1, padding = 1 })
  module:append(fn.batch_norm2d, { output = "block"..serial.."_0i" })
  if stride == 1 then
    module:append(fn.add       , { alpha  = "block"..(serial - 1) })
  else
    module:append(fn.conv2d    , { input  = "block"..(serial - 1),
    	output = "block"..serial.."_0t", stride = stride })
    module:append(fn.batch_norm2d)
    module:append(fn.add       , { alpha  = "block"..serial.."_0i" })
  end
  module:append(fn.relu        , { name   = "#NULL" })
  module:append(fn.alias       , { output = "block"..serial.."_0" })
  module:append(fn.conv2d      , { stride = 1, padding = 1 })
  module:append(fn.batch_norm2d)
  module:append(fn.relu        , { name   = "#NULL" })
  module:append(fn.conv2d      , { stride = 1, padding = 1 })
  module:append(fn.batch_norm2d, { output = "block"..serial.."_1" })
  module:append(fn.add         , { alpha  = "block"..serial.."_0" })
  module:append(fn.relu        , { name = "#NULL" })
  module:append(fn.alias       , { output = "block"..serial })
end

-- block0
features:append(fn.conv2d      , { input  = "#in", stride = 2, padding = 3 })
features:append(fn.batch_norm2d)
features:append(fn.relu        , { name   = "#NULL" })
features:append(fn.max_pool2d  , { kernel = 3, stride = 2, output = "block0" })

make_basic_block(features, 1, 1) -- block1
make_basic_block(features, 2, 2) -- block2
make_basic_block(features, 3, 2) -- block3
make_basic_block(features, 4, 2) -- block4

-- fc
 --[[ Replace the kernel-stride fixed pooling to GAP ]]
classifier:append(fn.avg_pool2d, { kernel = 7, stride = 7 })
classifier:append(fn.reshape, { shape = { -1, 1, 1 } }) -- C * H * W
classifier:append(fn.fully_connected)
classifier:append(fn.softmax, { output = "#*out", name = "#NULL"})

resnet18.block = "f" -- features
features:impl_on(resnet18)

resnet18.block = "c" -- classifier
classifier:impl_on(resnet18)

resnet18:generate()
