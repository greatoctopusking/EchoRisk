from pydicom import dcmread
ds = dcmread(r"D:\玛卡巴卡\MATERIALS\22spring\EchoRisk竞赛\Dataset\Task1\dicom\train\ECHORISK_0066\T5\ECHORISK_0066_T5_A2C.dcm")
arr = ds.pixel_array
print(arr.shape)