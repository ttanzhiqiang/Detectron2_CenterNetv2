import numpy
import os
import pickle
import re
import sys
from numpy import array

# Example: python ImportBaseline.py model_final_997cc7
# modelName = sys.argv[1]
modelName = 'model_final_f6e8b1'

# checkpoints = os.getenv('D2_CHECKPOINTS_DIR') + '\\'
checkpoints = 'D:\\libtorch\\detectron2_project\\Detectron2\\Import\\'

fcpp = open(os.getcwd() + '\\Baseline\\' + modelName + '.cpp', 'w')
fdataFileName = checkpoints + modelName + '.data'
fdata = open(fdataFileName, 'wb')
loaded = pickle.load(open(checkpoints + modelName + '.pkl', 'rb'))
model = loaded["model"]

fcpp.write('#include "Base.h"\n')
fcpp.write('#include <Detectron2/Import/ModelImporter.h>\n')
fcpp.write('\n')
fcpp.write('using namespace Detectron2;\n')
fcpp.write('\n')
fcpp.write('/' * 119)
fcpp.write('\n')
fcpp.write('\n')

fcpp.write('std::string ModelImporter::import_' + modelName + '() {\n')
offset = 0
num = 0
for key in model:
    m_key = key[0:18]
    m_key_1 = key[0:9]
    m_key_box_predictor = key[0:23]
    m_key_fpn = key[9:12]
    data = model[key]
    shape = data.shape
    numel = data.size
    data = data.reshape([numel])

    if (m_key_fpn == 'fpn'):
        continue
    if (m_key_box_predictor == 'roi_heads.box_predictor'):
        continue
    if (m_key_1 == 'roi_heads'):
        continue
    if (m_key == 'proposal_generator') :
        continue
    fcpp.write('\tAdd("' + key + '", ' + str(numel) + '); // ' + str(offset) + '\n')
    fdata.write(data.tobytes())
    offset += numel * 4
    assert fdata.tell() == offset, "{} != {}".format(fdata.tell(), offset)
    num = num+1

fcpp.write('\n')
fcpp.write('\treturn DataDir() + "\\\\' + modelName + '.data";\n')
fcpp.write('}\n')
