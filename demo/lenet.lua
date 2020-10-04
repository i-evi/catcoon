loadfile("./util/lua2cc.lua")() 

network = {
    networkName  = "lenet",
    createScope  = "lenet",
    parameterLv  = 0,
    inputLayers  = {"in"},
    outputLayers = {"out"},
    l1  = conv2d         ({input = "in", stride = 1, padding = 2}),
    l2  = relu           ({input = "l1"}),
    l3  = maxPool2d      ({input = "l2", stride = 2}),
    l4  = conv2d         ({input = "l3", stride = 1, padding = 2}),
    l5  = relu           ({input = "l4"}),
    l6  = maxPool2d      ({input = "l5", stride = 2}),
    l7  = reshape        ({input = "l6", shape = {-1, 1, 1}}),
    l8  = fullyConnected ({input = "l7"}),
    l9  = relu           ({input = "l8"}),
    l10 = fullyConnected ({input = "l9"}),
    out = softmax        ({input = "l10"})
}

ccCodeTranslator(network, {file = "_lenet.c"})
