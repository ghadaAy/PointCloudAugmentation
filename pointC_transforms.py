import torch
import math
import numpy as np


class RandomHorizontalFlip(torch.nn.Module):
  """Horizontally flip the given point cloud randomly with a given probability
    Args:
        p (float): probability of the point cloud being flipped. Default value is 0.5
  """

  def __init__(self, p=0.5):
      super().__init__()
      self.p = p

  def forward(self, x):
    ''' randomly flip pointcloud tensor'''
    if self.p>0.5:
      x[:,1] = x[:,1]*(-1)
    return x

  def __rep__(self)->str : 
    return f"self.__class_.__name__(p={self.p})"

class RandomVerticalFlip(torch.nn.Module):
  """Vertical flip the given point cloud randomly with a given probability
    Args:
        p (float): probability of the point cloud being flipped. Default value is 0.5
  """
  def __init__(self, p=0.5):
      super().__init__()
      self.p = p

  def forward(self, x):
    if self.p>0.5:
      x[:,0] = x[:,0]*(-1)
    return x
  
  def __rep__(self)->str : 
    return f"self.__class_.__name__(p={self.p})"


class TranslatePointCloud(torch.nn.Module):

  """Translates the given point cloud randomly with a given probability
    Args:
        trans (tuple of len 3): values by which we translate on each axis
  """
  def __init__(self, trans):
    super().__init__()
    assert len(trans)==3 , "you must pass 3 values in trans"
    assert type(trans)== tuple, "you must pass a tuple of size 3"
    self.trans = trans

  def forward(self, pcd):
    x=pcd.double()
    trans_m=torch.tensor([[1,0,0],[0,1,0],self.trans]).double()
    return x@trans_m
  def __rep__(self)->str : 
    return f"self.__class_.__name__(p={self.p})"


class ShearPointCloud(torch.nn.Module):

  """Shears the given point cloud randomly with the sinh of a given value
    Args:
        shear deg (float): value by its sinh we shear the pcd
  """
  def __init__(self, shear_deg):
    super().__init__()
    #assert len(trans)==3 , "you must pass 3 values in trans"
    #assert type(trans)== tuple, "you must pass a tuple of size 3"
    self.shear_deg = shear_deg

  def forward(self, pcd):
    Sinh = np.sinh(math.radians(self.shear_deg))

    x=pcd.double()
    shear_m=torch.tensor([[1,Sinh,Sinh],[Sinh,1,Sinh],[Sinh,Sinh,1]]).double()
    return x@shear_m

  def __rep__(self)->str : 
    return f"self.__class_.__name__(p={self.p})"

class RescalePointCloud(torch.nn.Module):

  """Shears the given point cloud randomly with the sinh of a given value
    Args:
        shear deg (float): value by its sinh we shear the pcd
  """
  def __init__(self, rescale):
    super().__init__()
    assert len(rescale)==3 , "you must pass 3 values in rescale"
    assert type(rescale)== tuple, "you must pass a tuple of size 3"
    self.rescale = rescale

  def forward(self, pcd):
    x=pcd.double()
    rescale_m=torch.tensor([[self.rescale[0],0,0],[0,self.rescale[1],0],[0,0,self.rescale[2]]]).double()
    return x@rescale_m

  def __rep__(self)->str : 
    return f"self.__class_.__name__(p={self.p})"

class RotatePointCloud(torch.nn.Module):

  """Shears the given point cloud randomly with the sinh of a given value
    Args:
        shear deg (float): value by its sinh we shear the pcd
  """
  def __init__(self, rot_deg):
    super().__init__()
    #assert len(rescale)==3 , "you must pass 3 values in rescale"
    #assert type(rescale)== tuple, "you must pass a tuple of size 3"
    self.rot_deg = rot_deg

  def forward(self, pcd):
    alpha = math.radians(self.rot_deg)
    cos = math.cos(alpha)
    sin = math.sin(alpha)

    a=pcd.double()
    b1=torch.tensor([[1,0,0],[0,cos,-sin],[0,sin,cos]]).double()
    pc1 = a@b1
    b2 = torch.tensor([[cos,0,sin],[0,1,0],[-sin,0,cos]]).double()
    pc2 = pc1@b2
    b3 = torch.tensor([[cos,-sin,0],[sin,cos,0],[0,0,1]]).double()
    rot_m = pc2@b3

    return rot_m

  def __rep__(self)->str : 
    return f"self.__class_.__name__(p={self.p})"

class NormalizePointCloud(torch.nn.Module):
    """Normalize a feature. By default, features will be scaled between [0,1]. Should only be applied on a dataset-level.

    Parameters
    ----------
    standardize: bool: Will use standardization rather than scaling.
    """

    def __init__(self):
        super().__init__()

    def forward(self, data):
        data[:,0] = (data[:,0] - data[:,0].min()) / (data[:,0].max() - data[:,0].min())
        data[:,1] = (data[:,1] - data[:,1].min()) / (data[:,1].max() - data[:,1].min())
        data[:,2] = (data[:,2] - data[:,2].min()) / (data[:,2].max() - data[:,2].min())
        return data

    def __repr__(self):
        return "{}(feature_name={}, standardize={})".format(self.__class__.__name__, self._feature_name, self._standardize)

