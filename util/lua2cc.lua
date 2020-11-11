common_attr = {
  name   = true,  -- name for the yield feature maps
  input  = true,
  output = true,
  shape  = true,
  template  = true,
  parameter = true,
  -- used for layers
  alpha   = true,
  beta    = true,
  gamma   = true,
  delta   = true,
  epsilon = true,
  zeta    = true,
  eta     = true,
  theta   = true,
  iota    = true,
  kappa   = true,
  lambda  = true,
  --[[ ... ]]
}

table.copy = function(src)
  local lt = {}
  local function _copy(src)
    if type(src) ~= "table" then
        return src
    elseif lt[src] then
        return lt[src]
    end
    local t = {}
    lt[src] = t
    for k, v in pairs(src) do
        t[_copy(k)] = _copy(v)
    end
    if getmetatable ~= nil then
      return setmetatable(t, getmetatable(src))
    else
      return t
    end
  end
  return _copy(src)
end

local fputs = function(fp, ...)
  local args = { ... }
  for k, v in pairs(args) do
    fp:write(v)
  end
end

-- _ctrlfp = io.open(cfg.file, "w+")
_ctrlfp = io.stdout
local print_line = function(line, indent)
  if indent == nil then indent = 0 end
  local lineLimit = 80
  local indentOff = indent * 8
  local indentStr = string.rep("\t", indent)
  local csr = 1
  local brk = 0
  local pos = 0
  local nextword = ""
  repeat
    if pos == 0 then
      pos = indentOff
      fputs(_ctrlfp, indentStr)
    end
    brk, _ = string.find(line, ',', csr)
    if brk ~= nil then
      nextword = string.sub(line, csr, brk)
    else
      nextword = string.sub(line, csr)
    end
    csr = csr + #nextword
    if pos + #nextword >= lineLimit then
      fputs(_ctrlfp, '\n');
      pos = indentOff
      fputs(_ctrlfp, indentStr)
    end
    if pos == indentOff then
      local off, _ = string.find(nextword, ' ')
      if off == 1 then
        nextword = string.sub(nextword, 2)
      end
    end
    fputs(_ctrlfp, string.format("%s", nextword))
    pos = pos + #nextword
  until csr >= #line
  fputs(_ctrlfp, '\n')
end

local custom_str_chk = function(str, prefix, ctx)
  if string.byte(str, 1) == 35 then -- '#'
    str = string.sub(str, 2)
    if ctx.outputs[str] ~= nil then
      str = '*'..str
    end
  else
    str = string.format("%s_%s", prefix , str)
  end
  return str
end

local seq_module = {
  buf_units = {},
  nunits = 0,
  append = function(self, cfg, arg)
    local reg = table.copy(cfg)
    if arg then
      for k, v in pairs(arg) do
        reg[k] = v
      end
    end
    table.insert(self.buf_units, reg)
    self.nunits = self.nunits + 1
  end,
  impl_on = function(self, ctx)
    local layernum = 0
    local ctx_name = ctx.name
    local prefix = ctx.block
    for _, cfg in pairs(self.buf_units) do
      local line = cfg.template
      -- check ctx.lastout
      if not ctx.lastout then
        assert(ctx.input_dfl, "Network should take at least one input")
        ctx.lastout = ctx.input_dfl
      end
      -- match $input
      local input = cfg.input
      if input == nil then
        input = ctx.lastout
      else
        input = custom_str_chk(input, prefix, ctx)
      end
      ctx.layer_list[input] = true -- reg $input
      line = string.gsub(line, "$input", input)
      -- match $output
      local output = cfg.output
      local fmtlcc = #tostring(self.nunits)
      local layernumfmt = "%s_%0"..fmtlcc.."d"
      if output == nil then
        output = string.format(layernumfmt, prefix, layernum)
        layernum = layernum + 1 -- update layernum
      else
        output = custom_str_chk(output, prefix, ctx)
      end
      line = string.gsub(line, "$output", output)
      -- strip `output` begin with '*'
      if string.byte(output, 1) == 42 then -- '*'
        output = string.sub(output, 2)
      end
      ctx.layer_list[output] = true -- reg $output
      -- update ctx.output
      ctx.lastout = output
      -- match name
      local name = cfg.name
      if name == nil then
        name = string.format("%s/%s", ctx_name, output)
      else
        if string.byte(name, 1) == 35 then -- '#'
          name = string.sub(name, 2)
        else
          name = string.format("%s/%s_%s", ctx_name, prefix, name)
        end
      end
      if name ~= "NULL" then
        name = string.format("\"%s\"", name)
      end
      line = string.gsub(line, "$name", name)
      -- match shape
      if cfg.shape then
        local shape = string.format(ctx.shapefmt, #ctx.shape_list)
        line = string.gsub(line, "$shape", shape)
        table.insert(ctx.shape_list, cfg.shape)
      end
      -- match layers, $alpha, $beta...
      if cfg.layers then
        for k, v in pairs(cfg.layers) do
          local layer = cfg[v]
          layer = custom_str_chk(layer, prefix, ctx)
          ctx.layer_list[layer] = true -- reg $layer
          line = string.gsub(line, string.format("$%s", v), layer)
        end
      end
      -- match custom args
      for k,v in pairs(cfg) do
        if not common_attr[k] then
          line = string.gsub(line, "$"..k, v)
        end
      end
      -- match parameters
      repeat
        local b, e = string.find(line, "$pack.(%a)")
        if not b then
          break
        end
        local para = string.sub(line, b, e)
        line = string.gsub(line, para, string.format(ctx.parafmt, ctx.paranum))
        ctx.paranum = ctx.paranum + 1
        para = string.gsub(para, "$pack", string.format(ctx.packfmt, ctx.packnum))
        table.insert(ctx.pack_list, para)
      until false
      if cfg.parameter then
        ctx.packnum = ctx.packnum + 1
      end
      table.insert(ctx.code_lines, line)
    end
  end
}

local context_dfl = {
  name  = "net",     -- context name
  block = "blk",     -- current block name
  lastout  = nil,    -- last output's name
  packnum  = 0,
  paranum  = 0,
  packfmt  = "%03d",
  parafmt  = "_p[%d]",
  shapefmt = "_shape%d",
  declaration = nil,
  inputs      = {},
  outputs     = {},
  code_lines  = {},
  pack_list   = {}, -- required packed parameters
  shape_list  = {},
  layer_list  = {}, -- create layers
  print = function(self, k)
    for k, v in pairs(self[k]) do
      print(k, v)
    end
  end,
  generate = function(self)
    print_line(self.declaration)
    print_line("{")
    -- make shapes
    if #self.shape_list >= 1 then
      local shape = "static int "
      for k, v in pairs(self.shape_list) do
        shape = shape..string.format(self.shapefmt.."[] = {", k - 1)
        for _, v in pairs(v) do
          shape = shape..tostring(v)..", "
        end
        shape = shape.."0}"
        if k ~= #self.shape_list then
          shape = shape..", "
        else
          shape = shape..";"
        end
      end
      print_line(shape, 1)
    end
    -- make layers
    local layer_flag = false
    local sorted_layers = {}
    for _ in pairs(self.layer_list) do
      layer_flag = true
    end
    if layer_flag then
      local layers = "cc_tensor_t "
      for k, _ in pairs(self.inputs) do
        self.layer_list[k] = nil
      end
      for k, _ in pairs(self.outputs) do
        self.layer_list[k] = nil
      end
      for k, _ in pairs(self.layer_list) do
        table.insert(sorted_layers, k)
      end
      table.sort(sorted_layers)
      for _, v in pairs(sorted_layers) do
        layers = layers..string.format("*%s, ", v)
      end
      layers = string.sub(layers, 1, -3)..";"
      print_line(layers, 1)
    end
    -- make packed parameters
    local para = "static const char *p_namels[] = {"
    print_line(para, 1)
    para = ""
    for k, v in pairs(self.pack_list) do
      para = para..string.format("\"%s\"", v)..", "
    end
    para = string.sub(para, 1, -3).."};"
    print_line(para, 1)
    print_line(string.format(string.format(
      "static cc_tensor_t *%s;", self.parafmt), #self.pack_list), 1)
    print_line("static int i;", 1)
    print_line(string.format("for (; i < %d; ++i) {", #self.pack_list), 1)
    print_line((string.gsub(self.parafmt, "%%d", "i"))..
      " = cc_tsrmgr_get(p_namels[i]);", 2)
    print_line("}", 1)
    -- print code_lines
    for k, v in pairs(self.code_lines) do
      print_line(v, 1)
    end
    print_line("}")
  end
}

new_module = function(name, input, output)
  return table.copy(seq_module)
end

new_context = function(cfg)
  local ctx = table.copy(context_dfl)
  if cfg then
    for k, v in pairs(cfg) do
      if k == "inputs" then      -- map inputs
        for _, name in pairs(v) do
          ctx.input_dfl = name
          ctx.inputs[name] = true
        end
      elseif k == "outputs" then -- map outputs
        for _, name in pairs(v) do
          ctx.outputs[name] = true
        end
      else
        ctx[k] = v
      end
    end
  end
  local decl = string.format("void %s(", ctx.name)
  -- make inputs
  for k, v in pairs(ctx.inputs) do
    decl = decl..string.format("cc_tensor_t *%s, ", k)
  end
  -- make outputs
  for k, v in pairs(ctx.outputs) do
    decl = decl..string.format("cc_tensor_t **%s, ", k)
  end
  decl = string.format("%s)", string.sub(decl, 1, -3))
  ctx.declaration = decl
  return ctx
end
