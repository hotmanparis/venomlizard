import torch

horse0 = torch.load("saved_tensor_overfit1/ridehorse0.pt")
horse1 = torch.load("ridehorse1.pt")
bike1 = torch.load("ridebike1.pt")
truck1 = torch.load("ridetruck1.pt")
car1 = torch.load("ridecar1.pt")
ride = torch.load("ride.pt")

a = horse0[0][0] - horse1[0][0]
torch.norm(a)

a = horse0[0][0] - bike1[0][0]
torch.norm(a)

a = horse0[0][0] - truck1[0][0]
torch.norm(a)

a = horse0[0][0] - car1[0][0]
torch.norm(a)

hydrant0 = torch.load("firehydrant0.pt")
hydrant1 = torch.load("firehydrant1.pt")
hydrant2 = torch.load("firehydrantt2.pt")[0][0]
hydrant3 = torch.load("firehydrant3.pt")

place0 = torch.load("fireplace0.pt")
place1 = torch.load("fireplace1.pt")
place2 = torch.load("fireplacee2.pt")[0][0]
place3 = torch.load("fireplacee3.pt")[0][0]

place3_20 = torch.load("fireplace3_20.pt")
hydrant3_20 = torch.load("firehydrant3_20.pt")
a = place3_20 - hydrant3_20
torch.norm(a)

a = horse0[0][0] - hydrant0[0][0]
torch.norm(a)

b = 0
a = hydrant0[0][0] - hydrant1[0][0]
b += torch.norm(a)
a = hydrant0[0][0] - hydrant2[0][0]
b += torch.norm(a)
a = hydrant0[0][0] - hydrant3[0][0]
b += torch.norm(a)
a = hydrant1[0][0] - hydrant2[0][0]
b += torch.norm(a)
a = hydrant1[0][0] - hydrant3[0][0]
b += torch.norm(a)
a = hydrant2[0][0] - hydrant3[0][0]
b += torch.norm(a)
print(b/6)

c = 0
a = place0[0][0] - place1[0][0]
c += torch.norm(a)
a = place0[0][0] - place2[0][0]
c += torch.norm(a)
a = place0[0][0] - place3[0][0]
c += torch.norm(a)
a = place1[0][0] - place2[0][0]
c += torch.norm(a)
a = place1[0][0] - place3[0][0]
c += torch.norm(a)
a = place2[0][0] - place3[0][0]
c += torch.norm(a)
print(c/6)

d = 0
a = hydrant0[0][0] - place0[0][0]
d += torch.norm(a)
a = hydrant0[0][0] - place1[0][0]
d += torch.norm(a)
a = hydrant0[0][0] - place2[0][0]
d += torch.norm(a)
a = hydrant0[0][0] - place3[0][0]
d += torch.norm(a)

a = hydrant1[0][0] - place0[0][0]
d += torch.norm(a)
a = hydrant1[0][0] - place1[0][0]
d += torch.norm(a)
a = hydrant1[0][0] - place2[0][0]
d += torch.norm(a)
a = hydrant1[0][0] - place3[0][0]
d += torch.norm(a)

a = hydrant2[0][0] - place0[0][0]
d += torch.norm(a)
a = hydrant2[0][0] - place1[0][0]
d += torch.norm(a)
a = hydrant2[0][0] - place2[0][0]
d += torch.norm(a)
a = hydrant2[0][0] - place3[0][0]
d += torch.norm(a)

a = hydrant3[0][0] - place0[0][0]
d += torch.norm(a)
a = hydrant3[0][0] - place1[0][0]
d += torch.norm(a)
a = hydrant3[0][0] - place2[0][0]
d += torch.norm(a)
a = hydrant3[0][0] - place3[0][0]
d += torch.norm(a)
print(d/16)