local parameterLv  = 0  -- PKG: Begin at 0
local shapeCounter = 0
local layerCounter = 1 -- MUST Begin at 1
local parameterCnt = 1 -- MUST Begin at 1
local staticSegBuf = {}
local layerOutputs = {}
local _ctrlfp = io.stdout

local makeLayer = function(t, paraNum)
  t.layerId = layerCounter
  layerCounter = layerCounter + 1
  if paraNum > 0 then
    t.paraLv = parameterLv
    parameterLv = parameterLv + 1
    t.paraId = parameterCnt
    parameterCnt = parameterCnt + paraNum
  end
end

conv2d = function(args)
  local ret = args
  makeLayer(ret, 2)
  ret.op = "conv2d"
  assert(ret.stride , "conv2d: \"stride\" not set")
  assert(ret.padding, "conv2d: \"padding\" not set")
  if not ret.offset then
    ret.offset = 0
  end
  -- output = conv2d(input, weight, bias, stride, padding, offset, name)
  ret.format = function(output, info, parals, scope)
    local name = info.name
    if not name then
      name = output
    end
    if type(scope) == "string" then
      name = string.format("%s/%s", scope, name)
    end
    parals[info.paraId + 0] = string.format("%03d.w", info.paraLv)
    parals[info.paraId + 1] = string.format("%03d.b", info.paraLv)
    layerOutputs[ret.layerId] = output
    if info.input == nil then
      if info.layerId - 1 < 1 then
        assert(nil, "must specify an input for the 1st layer")
      end
      info.input = string.format("@%d", info.layerId - 1)
    end
    return string.format(
      "%s = cc_conv2d(%s, __pls[%d], __pls[%d], %d, %d, %d, \"%s\");",
      output, info.input, info.paraId - 1, info.paraId, info.stride,
      info.padding, info.offset, name)
  end
  return ret
end

dwConv2d = function(args)
  local ret = args
  makeLayer(ret, 2)
  ret.op = "dwConv2d"
  assert(ret.stride , "dwConv2d: \"stride\" not set")
  assert(ret.padding, "dwConv2d: \"padding\" not set")
  if not ret.offset then
    ret.offset = 0
  end
  -- output = dwConv2d(input, weight, bias, stride, padding, offset, name)
  ret.format = function(output, info, parals, scope)
    local name = info.name
    if not name then
      name = output
    end
    if type(scope) == "string" then
      name = string.format("%s/%s", scope, name)
    end
    parals[info.paraId + 0] = string.format("%03d_w", info.paraLv)
    parals[info.paraId + 1] = string.format("%03d_b", info.paraLv)
    layerOutputs[ret.layerId] = output
    if info.input == nil then
      if info.layerId - 1 < 1 then
        assert(nil, "must specify an input for the 1st layer")
      end
      info.input = string.format("@%d", info.layerId - 1)
    end
    return string.format(
      "%s = cc_conv2d(%s, __pls[%d], __pls[%d], %d, %d, %d, \"%s\");",
      output, info.input, info.paraId - 1, info.paraId, info.stride,
      info.padding, info.offset, name)
  end
  return ret
end

pwConv2d = function(args)
  local ret = args
  makeLayer(ret, 2)
  ret.op = "pwConv2d"
  -- output = pwConv2d(input, weight, bias, name)
  ret.format = function(output, info, parals, scope)
    local name = info.name
    if not name then
      name = output
    end
    if type(scope) == "string" then
      name = string.format("%s/%s", scope, name)
    end
    parals[info.paraId + 0] = string.format("%03d_w", info.paraLv)
    parals[info.paraId + 1] = string.format("%03d_b", info.paraLv)
    layerOutputs[ret.layerId] = output
    if info.input == nil then
      if info.layerId - 1 < 1 then
        assert(nil, "must specify an input for the 1st layer")
      end
      info.input = string.format("@%d", info.layerId - 1)
    end
    return string.format(
      "%s = cc_pw_conv2d(%s, __pls[%d], __pls[%d], \"%s\");",
      output, info.input, info.paraId - 1, info.paraId, name)
  end
  return ret
end

fullyConnected = function(args)
  local ret = args
  makeLayer(ret, 2)
  ret.op = "fullyConnected"
  -- output = fullyConnected(input, weight, bias, name)
  ret.format = function(output, info, parals, scope)
    local name = info.name
    if not name then
      name = output
    end
    if type(scope) == "string" then
      name = string.format("%s/%s", scope, name)
    end
    parals[info.paraId + 0] = string.format("%03d_w", info.paraLv)
    parals[info.paraId + 1] = string.format("%03d_b", info.paraLv)
    layerOutputs[ret.layerId] = output
    if info.input == nil then
      if info.layerId - 1 < 1 then
        assert(nil, "must specify an input for the 1st layer")
      end
      info.input = string.format("@%d", info.layerId - 1)
    end
    return string.format(
      "%s = cc_fully_connected(%s, __pls[%d], __pls[%d], \"%s\");",
      output, info.input, info.paraId - 1, info.paraId, name)
  end
  return ret
end

relu = function(args)
  local ret = args
  makeLayer(ret, 0)
  ret.op = "relu"
  --[[ output = relu(input, name) ]]
  ret.format = function(output, info, parals, scope)
    local name = info.name
    if not name then
      name = "NULL"
    else
      if type(scope) == "string" then
        name = string.format("%s/%s", scope, name)
      end
      name = string.format("\"%s\"", name)
    end
    layerOutputs[ret.layerId] = output
    if info.input == nil then
      if info.layerId - 1 < 1 then
        assert(nil, "must specify an input for the 1st layer")
      end
      info.input = string.format("@%d", info.layerId - 1)
    end
    return string.format(
      "%s = cc_relu(%s, %s);",
      output, info.input, name)
  end
  return ret
end

softmax = function(args)
  local ret = args
  makeLayer(ret, 0)
  ret.op = "softmax"
  --[[ output = softmax(input, name) ]]
  ret.format = function(output, info, parals, scope)
    local name = info.name
    if not name then
      name = "NULL"
    else
      if type(scope) == "string" then
        name = string.format("%s/%s", scope, name)
      end
      name = string.format("\"%s\"", name)
    end
    layerOutputs[ret.layerId] = output
    if info.input == nil then
      if info.layerId - 1 < 1 then
        assert(nil, "must specify an input for the 1st layer")
      end
      info.input = string.format("@%d", info.layerId - 1)
    end
    return string.format(
      "%s = cc_softmax(%s, %s);",
      output, info.input, name)
  end
  return ret
end

maxPool2d = function(args)
  local ret = args
  makeLayer(ret, 0)
  ret.op = "maxPool2d"
  assert(ret.stride , "maxPool2d: \"stride\" not set")
  --[[ output = relu(input, name) ]]
  ret.format = function(output, info, parals, scope)
    local name = info.name
    if not name then
      name = output
    end
    if type(scope) == "string" then
      name = string.format("%s/%s", scope, name)
    end
    layerOutputs[ret.layerId] = output
    if info.input == nil then
      if info.layerId - 1 < 1 then
        assert(nil, "must specify an input for the 1st layer")
      end
      info.input = string.format("@%d", info.layerId - 1)
    end
    return string.format("%s = cc_max_pool2d(%s, %d, \"%s\");",
      output, info.input, info.stride, name)
  end
  return ret
end

batchNorm2d = function(args)
  local ret = args
  makeLayer(ret, 1)
  ret.op = "batchNorm2d"
  --[[ output = relu(input, name) ]]
  ret.format = function(output, info, parals, scope)
    local name = info.name
    if not name then
      name = output
    end
    if type(scope) == "string" then
      name = string.format("%s/%s", scope, name)
    end
    parals[info.paraId] = string.format("%03d_n", info.paraLv)
    layerOutputs[ret.layerId] = output
    if info.input == nil then
      if info.layerId - 1 < 1 then
        assert(nil, "must specify an input for the 1st layer")
      end
      info.input = string.format("@%d", info.layerId - 1)
    end
    return string.format(
      "%s = cc_batch_norm(%s, __pls[%d], \"%s\");",
      output, info.input, info.paraId - 1, name)
  end
  return ret
end

reshape = function(args)
  local ret = args
  makeLayer(ret, 0)
  ret.op = "reshape"
  assert(ret.shape, "reshape: \"shape\" is required")
  local buf = string.format(
    "static int __shape%d[] = {", shapeCounter)
  for k, v in pairs(ret.shape) do
    buf = string.format("%s%d, ", buf, v)
  end
  buf = string.format("%s0};", buf, v)
  table.insert(staticSegBuf, buf)
  ret.shapeId = shapeCounter
  shapeCounter = shapeCounter + 1
  ret.format = function(output, info, parals, scope)
    if info.input == nil then
      if info.layerId - 1 < 1 then
        assert(nil, "must specify an input for the 1st layer")
      end
      info.input = layerOutputs[info.layerId - 1]
    end
    local code = string.format(
      "%s = cc_tensor_reshape(%s, __shape%d);",
      output, info.input, info.shapeId)
    layerOutputs[ret.layerId] = output
    return code
  end
  return ret
end

local fprint = function(fp, ...)
  local args = { ... }
  local flag = false
  for k, v in pairs(args) do
    if not flag then
      flag = true
    else
      fp:write('\t')
    end
    fp:write(v)
  end
  fp:write('\n')
end

local printLine = function(line, indent)
  if indent == nil then indent = 0 end
  local indentOff = indent * 8
  local llen = #line + indentOff
  local indentStr = string.rep("\t", indent)
  line = indentStr..line
  if llen <= 80 then
    fprint(_ctrlfp, line)
    return
  end
  local prev = 0
  local curr = 0
  repeat
    curr, _ = string.find(line, ',', prev + 1)
    if curr == nil then
      break
    end
    if (curr + indentOff) >= 80 then
      local buf = string.sub(line, 1, prev)
      fprint(_ctrlfp, buf)
      line = string.sub(line, prev + 1)
      if string.byte(line, 1) == 32 then
        line = string.sub(line, 2)
      end
    end
    prev = curr
  until (#line + indentOff) < 80
  if #line > 80 then
    local looking = string.byte(',', 1)
    for i = #line, 60, -1 do
      if string.byte(line, i) == looking then
        fprint(_ctrlfp, string.sub(line, 1, i))
        line = string.sub(line, i + 1)
        break
      end
    end
  end
  curr, _ = string.find(line, ' ')
  if curr == 1 then
    line = string.sub(line, 2)
  end
  line = indentStr..line
  fprint(_ctrlfp, line)
end

local runningFlag = true

ccCodeTranslator = function(net, cfg)
  assert(runningFlag,
    "This beta version can only handle 1 network")
  runningFlag = false
  local createTsr = {}
  local keyFilter = {}
  local codeLines = {}
  local paraxList = {}
  local indentOff = 1
  local keyWords = {inputLayers = true, outputLayers = true}

  if type(cfg) == "table" then
    if type(cfg.file) == "string" then
      _ctrlfp = io.open(cfg.file, "w+")
      assert(_ctrlfp,
        string.format("failed open file: [%s]", cfg.file))
    end
  end

  local netName = net.networkName
  assert(netName, "Network's name not set")

  local netDef = string.format("void %s(", netName)
  for k, v in pairs(net.inputLayers) do
    netDef = netDef..string.format("cc_tensor_t *%s, ", v)
    keyFilter[v] = true
  end
  for k, v in pairs(net.outputLayers) do
    netDef = netDef..string.format("cc_tensor_t **%s, ", v)
    keyFilter[v] = true
  end
  netDef = string.sub(netDef, 1, #netDef - 2)..")"
  printLine(netDef, 0)
  printLine("{", 0)
  
  for k, v in pairs(staticSegBuf) do
    printLine(v, 1)
  end

  for k, v in pairs(net) do
    repeat
      if keyWords[k] then break end
      if type(v) == "table" then
        if net.parameterLv ~= nil and v.paraLv ~= nil then
          v.paraLv = v.paraLv + net.parameterLv
        end
        if keyFilter[k] then
          k = string.format("*%s", k)
        else
          createTsr[v.layerId] = k
        end
        codeLines[v.layerId] = v.format(k, v, paraxList, net.createScope)
      end
      break
    until true
  end

  local paraDef = "static const char *p_namels[] = {"
  printLine(paraDef, indentOff + 0)
  paraDef = ""
  for k,v in pairs(paraxList) do
    paraDef = paraDef.."\""..v.."\", "
  end
  paraDef = string.sub(paraDef, 1, #paraDef - 2).."};"
  printLine(paraDef, indentOff + 1)
  printLine(string.format(
    "static cc_tensor_t *__pls[%d];", #paraxList), indentOff + 0)

  local layerDef = "cc_tensor_t "
  for k, v in pairs(createTsr) do
    layerDef = layerDef..string.format("*%s, ", v)
  end
  layerDef = string.sub(layerDef, 1, #layerDef - 2)..";"
  printLine(layerDef, indentOff + 0)

  printLine("static int i;", indentOff + 0)
  printLine(string.format(
    "for (; i < %d; ++i) {", #paraxList), indentOff + 0)
  printLine(string.format(
    "__pls[i] = cc_tsrmgr_get(p_namels[i]);"), indentOff + 1)
  printLine("}", indentOff + 0)

  for k, v in pairs(codeLines) do
    v = string.gsub(v, "@%d*,",
      function(s)
        return string.format("%s,",
          layerOutputs[tonumber(string.sub(s, 2, #s - 1))])
      end)
    printLine(v, indentOff + 0)
  end
  printLine("}", 0)
  if _ctrlfp ~= io.stdout then
    io.close(_ctrlfp)
    _ctrlfp = io.stdout
  end
end
