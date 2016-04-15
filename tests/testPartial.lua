--[[

Test partial reading.

]]

require 'hdf5'

local totem = require 'totem'
local tester = totem.Tester()
local myTests = {}

local file_path = 'tests/data/twoTensors.h5'

function myTests:testRangeDeduction()
    local file = hdf5.open(file_path)
    local data = file:read('data1')

    local v1 = data:partial({1, 1})
    tester:assert(v1:nDimension() == 2 and v1:size(1) == 1 and v1:size(2) == 10)
    local v2 = data:partial({1, 2})
    tester:assert(v2:nDimension() == 2 and v2:size(1) == 2 and v2:size(2) == 10)
    local v3 = data:partial({1, 10})
    tester:assert(v3:nDimension() == 2 and v3:size(1) == 10 and v2:size(2) == 10)

    tester:assertError(function() data:partial({0, 1}) end)
    tester:assertError(function() data:partial({1, 11}) end)
    file:close()
end

function myTests:testUserSuppliedTensor()
    local file = hdf5.open(file_path)
    local data = file:read('data1')
    local t = torch.FloatTensor(2, 10)

    local v1 = data:partial(t, {1, 2})
    tester:assert(v1:data() == t:data())
    local v2 = data:partial(t, {2, 3})
    tester:assert(v2:data() == t:data())
    local v3 = data:partial(t, {2, 3}, {1, 10})
    tester:assert(v3:data() == t:data())

    tester:assertError(function() data:partial(t) end)
    tester:assertError(function() data:partial(t, {1, 3}) end)
    file:close()
end

return tester:add(myTests):run()
