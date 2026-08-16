import pandapower as pp
import pandapower.networks as pn

# 加载一个内置的IEEE 9节点标准案例
net = pn.case9()

# 运行潮流计算
pp.runpp(net)

print("✅ 潮流计算成功!")
print("\n各节点电压(标幺值):")
print(net.res_bus[["vm_pu", "va_degree"]])
print("\n各支路潮流(MW):")
print(net.res_line[["p_from_mw", "loading_percent"]])
