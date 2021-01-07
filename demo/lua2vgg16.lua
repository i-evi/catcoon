loadfile("./util/lua2cc.lua")()
loadfile("./util/ccfn.lua")()

vgg16 = new_context({
  name = "vgg16",
  inputs = {
    "in"
  },
  outputs = {
    "out"
  }
})

features   = new_module()
classifier = new_module()

for i = 1, 2 do
  if i == 1 then inp = "#in" else inp = nil end
  features:append(fn.conv2d    , { input = inp, stride = 1, padding = 1 })
  features:append(fn.relu      , { name = "#NULL"})
  features:append(fn.conv2d    , { input = nil, stride = 1, padding = 1 })
  features:append(fn.relu      , { name = "#NULL"})
  features:append(fn.max_pool2d, { kernel = 2, stride = 2 })
end

for i = 1, 3 do
  if i == 1 then inp = "in" else inp = nil end
  features:append(fn.conv2d    , { stride = 1, padding = 1 })
  features:append(fn.relu      , { name = "#NULL"})
  features:append(fn.conv2d    , { stride = 1, padding = 1 })
  features:append(fn.relu      , { name = "#NULL"})
  features:append(fn.conv2d    , { stride = 1, padding = 1 })
  features:append(fn.relu      , { name = "#NULL"})
  features:append(fn.max_pool2d, { kernel = 2, stride = 2 })
end

classifier:append(fn.reshape, { shape = { -1, 1, 1 } }) -- C * H * W
classifier:append(fn.fully_connected)
classifier:append(fn.relu, { name = "#NULL"})
classifier:append(fn.fully_connected)
classifier:append(fn.relu, { name = "#NULL"})
classifier:append(fn.fully_connected)
classifier:append(fn.softmax, { output = "#*out", name = "#NULL"})

vgg16.block = "f" -- features
features:impl_on(vgg16)

vgg16.block = "c" -- classifier
classifier:impl_on(vgg16)

vgg16:generate()
