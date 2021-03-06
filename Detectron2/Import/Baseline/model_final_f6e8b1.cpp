#include "Base.h"
#include <Detectron2/Import/ModelImporter.h>

using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string ModelImporter::import_model_final_f6e8b1() {
	Add("backbone.bottom_up.stem.conv1.weight", 9408); // 0
	Add("backbone.bottom_up.stem.conv1.norm.weight", 64); // 37632
	Add("backbone.bottom_up.stem.conv1.norm.bias", 64); // 37888
	Add("backbone.bottom_up.stem.conv1.norm.running_mean", 64); // 38144
	Add("backbone.bottom_up.stem.conv1.norm.running_var", 64); // 38400
	Add("backbone.bottom_up.res2.0.shortcut.weight", 16384); // 38656
	Add("backbone.bottom_up.res2.0.shortcut.norm.weight", 256); // 104192
	Add("backbone.bottom_up.res2.0.shortcut.norm.bias", 256); // 105216
	Add("backbone.bottom_up.res2.0.shortcut.norm.running_mean", 256); // 106240
	Add("backbone.bottom_up.res2.0.shortcut.norm.running_var", 256); // 107264
	Add("backbone.bottom_up.res2.0.conv1.weight", 4096); // 108288
	Add("backbone.bottom_up.res2.0.conv1.norm.weight", 64); // 124672
	Add("backbone.bottom_up.res2.0.conv1.norm.bias", 64); // 124928
	Add("backbone.bottom_up.res2.0.conv1.norm.running_mean", 64); // 125184
	Add("backbone.bottom_up.res2.0.conv1.norm.running_var", 64); // 125440
	Add("backbone.bottom_up.res2.0.conv2.weight", 36864); // 125696
	Add("backbone.bottom_up.res2.0.conv2.norm.weight", 64); // 273152
	Add("backbone.bottom_up.res2.0.conv2.norm.bias", 64); // 273408
	Add("backbone.bottom_up.res2.0.conv2.norm.running_mean", 64); // 273664
	Add("backbone.bottom_up.res2.0.conv2.norm.running_var", 64); // 273920
	Add("backbone.bottom_up.res2.0.conv3.weight", 16384); // 274176
	Add("backbone.bottom_up.res2.0.conv3.norm.weight", 256); // 339712
	Add("backbone.bottom_up.res2.0.conv3.norm.bias", 256); // 340736
	Add("backbone.bottom_up.res2.0.conv3.norm.running_mean", 256); // 341760
	Add("backbone.bottom_up.res2.0.conv3.norm.running_var", 256); // 342784
	Add("backbone.bottom_up.res2.1.conv1.weight", 16384); // 343808
	Add("backbone.bottom_up.res2.1.conv1.norm.weight", 64); // 409344
	Add("backbone.bottom_up.res2.1.conv1.norm.bias", 64); // 409600
	Add("backbone.bottom_up.res2.1.conv1.norm.running_mean", 64); // 409856
	Add("backbone.bottom_up.res2.1.conv1.norm.running_var", 64); // 410112
	Add("backbone.bottom_up.res2.1.conv2.weight", 36864); // 410368
	Add("backbone.bottom_up.res2.1.conv2.norm.weight", 64); // 557824
	Add("backbone.bottom_up.res2.1.conv2.norm.bias", 64); // 558080
	Add("backbone.bottom_up.res2.1.conv2.norm.running_mean", 64); // 558336
	Add("backbone.bottom_up.res2.1.conv2.norm.running_var", 64); // 558592
	Add("backbone.bottom_up.res2.1.conv3.weight", 16384); // 558848
	Add("backbone.bottom_up.res2.1.conv3.norm.weight", 256); // 624384
	Add("backbone.bottom_up.res2.1.conv3.norm.bias", 256); // 625408
	Add("backbone.bottom_up.res2.1.conv3.norm.running_mean", 256); // 626432
	Add("backbone.bottom_up.res2.1.conv3.norm.running_var", 256); // 627456
	Add("backbone.bottom_up.res2.2.conv1.weight", 16384); // 628480
	Add("backbone.bottom_up.res2.2.conv1.norm.weight", 64); // 694016
	Add("backbone.bottom_up.res2.2.conv1.norm.bias", 64); // 694272
	Add("backbone.bottom_up.res2.2.conv1.norm.running_mean", 64); // 694528
	Add("backbone.bottom_up.res2.2.conv1.norm.running_var", 64); // 694784
	Add("backbone.bottom_up.res2.2.conv2.weight", 36864); // 695040
	Add("backbone.bottom_up.res2.2.conv2.norm.weight", 64); // 842496
	Add("backbone.bottom_up.res2.2.conv2.norm.bias", 64); // 842752
	Add("backbone.bottom_up.res2.2.conv2.norm.running_mean", 64); // 843008
	Add("backbone.bottom_up.res2.2.conv2.norm.running_var", 64); // 843264
	Add("backbone.bottom_up.res2.2.conv3.weight", 16384); // 843520
	Add("backbone.bottom_up.res2.2.conv3.norm.weight", 256); // 909056
	Add("backbone.bottom_up.res2.2.conv3.norm.bias", 256); // 910080
	Add("backbone.bottom_up.res2.2.conv3.norm.running_mean", 256); // 911104
	Add("backbone.bottom_up.res2.2.conv3.norm.running_var", 256); // 912128
	Add("backbone.bottom_up.res3.0.shortcut.weight", 131072); // 913152
	Add("backbone.bottom_up.res3.0.shortcut.norm.weight", 512); // 1437440
	Add("backbone.bottom_up.res3.0.shortcut.norm.bias", 512); // 1439488
	Add("backbone.bottom_up.res3.0.shortcut.norm.running_mean", 512); // 1441536
	Add("backbone.bottom_up.res3.0.shortcut.norm.running_var", 512); // 1443584
	Add("backbone.bottom_up.res3.0.conv1.weight", 32768); // 1445632
	Add("backbone.bottom_up.res3.0.conv1.norm.weight", 128); // 1576704
	Add("backbone.bottom_up.res3.0.conv1.norm.bias", 128); // 1577216
	Add("backbone.bottom_up.res3.0.conv1.norm.running_mean", 128); // 1577728
	Add("backbone.bottom_up.res3.0.conv1.norm.running_var", 128); // 1578240
	Add("backbone.bottom_up.res3.0.conv2.weight", 147456); // 1578752
	Add("backbone.bottom_up.res3.0.conv2.norm.weight", 128); // 2168576
	Add("backbone.bottom_up.res3.0.conv2.norm.bias", 128); // 2169088
	Add("backbone.bottom_up.res3.0.conv2.norm.running_mean", 128); // 2169600
	Add("backbone.bottom_up.res3.0.conv2.norm.running_var", 128); // 2170112
	Add("backbone.bottom_up.res3.0.conv3.weight", 65536); // 2170624
	Add("backbone.bottom_up.res3.0.conv3.norm.weight", 512); // 2432768
	Add("backbone.bottom_up.res3.0.conv3.norm.bias", 512); // 2434816
	Add("backbone.bottom_up.res3.0.conv3.norm.running_mean", 512); // 2436864
	Add("backbone.bottom_up.res3.0.conv3.norm.running_var", 512); // 2438912
	Add("backbone.bottom_up.res3.1.conv1.weight", 65536); // 2440960
	Add("backbone.bottom_up.res3.1.conv1.norm.weight", 128); // 2703104
	Add("backbone.bottom_up.res3.1.conv1.norm.bias", 128); // 2703616
	Add("backbone.bottom_up.res3.1.conv1.norm.running_mean", 128); // 2704128
	Add("backbone.bottom_up.res3.1.conv1.norm.running_var", 128); // 2704640
	Add("backbone.bottom_up.res3.1.conv2.weight", 147456); // 2705152
	Add("backbone.bottom_up.res3.1.conv2.norm.weight", 128); // 3294976
	Add("backbone.bottom_up.res3.1.conv2.norm.bias", 128); // 3295488
	Add("backbone.bottom_up.res3.1.conv2.norm.running_mean", 128); // 3296000
	Add("backbone.bottom_up.res3.1.conv2.norm.running_var", 128); // 3296512
	Add("backbone.bottom_up.res3.1.conv3.weight", 65536); // 3297024
	Add("backbone.bottom_up.res3.1.conv3.norm.weight", 512); // 3559168
	Add("backbone.bottom_up.res3.1.conv3.norm.bias", 512); // 3561216
	Add("backbone.bottom_up.res3.1.conv3.norm.running_mean", 512); // 3563264
	Add("backbone.bottom_up.res3.1.conv3.norm.running_var", 512); // 3565312
	Add("backbone.bottom_up.res3.2.conv1.weight", 65536); // 3567360
	Add("backbone.bottom_up.res3.2.conv1.norm.weight", 128); // 3829504
	Add("backbone.bottom_up.res3.2.conv1.norm.bias", 128); // 3830016
	Add("backbone.bottom_up.res3.2.conv1.norm.running_mean", 128); // 3830528
	Add("backbone.bottom_up.res3.2.conv1.norm.running_var", 128); // 3831040
	Add("backbone.bottom_up.res3.2.conv2.weight", 147456); // 3831552
	Add("backbone.bottom_up.res3.2.conv2.norm.weight", 128); // 4421376
	Add("backbone.bottom_up.res3.2.conv2.norm.bias", 128); // 4421888
	Add("backbone.bottom_up.res3.2.conv2.norm.running_mean", 128); // 4422400
	Add("backbone.bottom_up.res3.2.conv2.norm.running_var", 128); // 4422912
	Add("backbone.bottom_up.res3.2.conv3.weight", 65536); // 4423424
	Add("backbone.bottom_up.res3.2.conv3.norm.weight", 512); // 4685568
	Add("backbone.bottom_up.res3.2.conv3.norm.bias", 512); // 4687616
	Add("backbone.bottom_up.res3.2.conv3.norm.running_mean", 512); // 4689664
	Add("backbone.bottom_up.res3.2.conv3.norm.running_var", 512); // 4691712
	Add("backbone.bottom_up.res3.3.conv1.weight", 65536); // 4693760
	Add("backbone.bottom_up.res3.3.conv1.norm.weight", 128); // 4955904
	Add("backbone.bottom_up.res3.3.conv1.norm.bias", 128); // 4956416
	Add("backbone.bottom_up.res3.3.conv1.norm.running_mean", 128); // 4956928
	Add("backbone.bottom_up.res3.3.conv1.norm.running_var", 128); // 4957440
	Add("backbone.bottom_up.res3.3.conv2.weight", 147456); // 4957952
	Add("backbone.bottom_up.res3.3.conv2.norm.weight", 128); // 5547776
	Add("backbone.bottom_up.res3.3.conv2.norm.bias", 128); // 5548288
	Add("backbone.bottom_up.res3.3.conv2.norm.running_mean", 128); // 5548800
	Add("backbone.bottom_up.res3.3.conv2.norm.running_var", 128); // 5549312
	Add("backbone.bottom_up.res3.3.conv3.weight", 65536); // 5549824
	Add("backbone.bottom_up.res3.3.conv3.norm.weight", 512); // 5811968
	Add("backbone.bottom_up.res3.3.conv3.norm.bias", 512); // 5814016
	Add("backbone.bottom_up.res3.3.conv3.norm.running_mean", 512); // 5816064
	Add("backbone.bottom_up.res3.3.conv3.norm.running_var", 512); // 5818112
	Add("backbone.bottom_up.res4.0.shortcut.weight", 524288); // 5820160
	Add("backbone.bottom_up.res4.0.shortcut.norm.weight", 1024); // 7917312
	Add("backbone.bottom_up.res4.0.shortcut.norm.bias", 1024); // 7921408
	Add("backbone.bottom_up.res4.0.shortcut.norm.running_mean", 1024); // 7925504
	Add("backbone.bottom_up.res4.0.shortcut.norm.running_var", 1024); // 7929600
	Add("backbone.bottom_up.res4.0.conv1.weight", 131072); // 7933696
	Add("backbone.bottom_up.res4.0.conv1.norm.weight", 256); // 8457984
	Add("backbone.bottom_up.res4.0.conv1.norm.bias", 256); // 8459008
	Add("backbone.bottom_up.res4.0.conv1.norm.running_mean", 256); // 8460032
	Add("backbone.bottom_up.res4.0.conv1.norm.running_var", 256); // 8461056
	Add("backbone.bottom_up.res4.0.conv2.weight", 589824); // 8462080
	Add("backbone.bottom_up.res4.0.conv2.norm.weight", 256); // 10821376
	Add("backbone.bottom_up.res4.0.conv2.norm.bias", 256); // 10822400
	Add("backbone.bottom_up.res4.0.conv2.norm.running_mean", 256); // 10823424
	Add("backbone.bottom_up.res4.0.conv2.norm.running_var", 256); // 10824448
	Add("backbone.bottom_up.res4.0.conv3.weight", 262144); // 10825472
	Add("backbone.bottom_up.res4.0.conv3.norm.weight", 1024); // 11874048
	Add("backbone.bottom_up.res4.0.conv3.norm.bias", 1024); // 11878144
	Add("backbone.bottom_up.res4.0.conv3.norm.running_mean", 1024); // 11882240
	Add("backbone.bottom_up.res4.0.conv3.norm.running_var", 1024); // 11886336
	Add("backbone.bottom_up.res4.1.conv1.weight", 262144); // 11890432
	Add("backbone.bottom_up.res4.1.conv1.norm.weight", 256); // 12939008
	Add("backbone.bottom_up.res4.1.conv1.norm.bias", 256); // 12940032
	Add("backbone.bottom_up.res4.1.conv1.norm.running_mean", 256); // 12941056
	Add("backbone.bottom_up.res4.1.conv1.norm.running_var", 256); // 12942080
	Add("backbone.bottom_up.res4.1.conv2.weight", 589824); // 12943104
	Add("backbone.bottom_up.res4.1.conv2.norm.weight", 256); // 15302400
	Add("backbone.bottom_up.res4.1.conv2.norm.bias", 256); // 15303424
	Add("backbone.bottom_up.res4.1.conv2.norm.running_mean", 256); // 15304448
	Add("backbone.bottom_up.res4.1.conv2.norm.running_var", 256); // 15305472
	Add("backbone.bottom_up.res4.1.conv3.weight", 262144); // 15306496
	Add("backbone.bottom_up.res4.1.conv3.norm.weight", 1024); // 16355072
	Add("backbone.bottom_up.res4.1.conv3.norm.bias", 1024); // 16359168
	Add("backbone.bottom_up.res4.1.conv3.norm.running_mean", 1024); // 16363264
	Add("backbone.bottom_up.res4.1.conv3.norm.running_var", 1024); // 16367360
	Add("backbone.bottom_up.res4.2.conv1.weight", 262144); // 16371456
	Add("backbone.bottom_up.res4.2.conv1.norm.weight", 256); // 17420032
	Add("backbone.bottom_up.res4.2.conv1.norm.bias", 256); // 17421056
	Add("backbone.bottom_up.res4.2.conv1.norm.running_mean", 256); // 17422080
	Add("backbone.bottom_up.res4.2.conv1.norm.running_var", 256); // 17423104
	Add("backbone.bottom_up.res4.2.conv2.weight", 589824); // 17424128
	Add("backbone.bottom_up.res4.2.conv2.norm.weight", 256); // 19783424
	Add("backbone.bottom_up.res4.2.conv2.norm.bias", 256); // 19784448
	Add("backbone.bottom_up.res4.2.conv2.norm.running_mean", 256); // 19785472
	Add("backbone.bottom_up.res4.2.conv2.norm.running_var", 256); // 19786496
	Add("backbone.bottom_up.res4.2.conv3.weight", 262144); // 19787520
	Add("backbone.bottom_up.res4.2.conv3.norm.weight", 1024); // 20836096
	Add("backbone.bottom_up.res4.2.conv3.norm.bias", 1024); // 20840192
	Add("backbone.bottom_up.res4.2.conv3.norm.running_mean", 1024); // 20844288
	Add("backbone.bottom_up.res4.2.conv3.norm.running_var", 1024); // 20848384
	Add("backbone.bottom_up.res4.3.conv1.weight", 262144); // 20852480
	Add("backbone.bottom_up.res4.3.conv1.norm.weight", 256); // 21901056
	Add("backbone.bottom_up.res4.3.conv1.norm.bias", 256); // 21902080
	Add("backbone.bottom_up.res4.3.conv1.norm.running_mean", 256); // 21903104
	Add("backbone.bottom_up.res4.3.conv1.norm.running_var", 256); // 21904128
	Add("backbone.bottom_up.res4.3.conv2.weight", 589824); // 21905152
	Add("backbone.bottom_up.res4.3.conv2.norm.weight", 256); // 24264448
	Add("backbone.bottom_up.res4.3.conv2.norm.bias", 256); // 24265472
	Add("backbone.bottom_up.res4.3.conv2.norm.running_mean", 256); // 24266496
	Add("backbone.bottom_up.res4.3.conv2.norm.running_var", 256); // 24267520
	Add("backbone.bottom_up.res4.3.conv3.weight", 262144); // 24268544
	Add("backbone.bottom_up.res4.3.conv3.norm.weight", 1024); // 25317120
	Add("backbone.bottom_up.res4.3.conv3.norm.bias", 1024); // 25321216
	Add("backbone.bottom_up.res4.3.conv3.norm.running_mean", 1024); // 25325312
	Add("backbone.bottom_up.res4.3.conv3.norm.running_var", 1024); // 25329408
	Add("backbone.bottom_up.res4.4.conv1.weight", 262144); // 25333504
	Add("backbone.bottom_up.res4.4.conv1.norm.weight", 256); // 26382080
	Add("backbone.bottom_up.res4.4.conv1.norm.bias", 256); // 26383104
	Add("backbone.bottom_up.res4.4.conv1.norm.running_mean", 256); // 26384128
	Add("backbone.bottom_up.res4.4.conv1.norm.running_var", 256); // 26385152
	Add("backbone.bottom_up.res4.4.conv2.weight", 589824); // 26386176
	Add("backbone.bottom_up.res4.4.conv2.norm.weight", 256); // 28745472
	Add("backbone.bottom_up.res4.4.conv2.norm.bias", 256); // 28746496
	Add("backbone.bottom_up.res4.4.conv2.norm.running_mean", 256); // 28747520
	Add("backbone.bottom_up.res4.4.conv2.norm.running_var", 256); // 28748544
	Add("backbone.bottom_up.res4.4.conv3.weight", 262144); // 28749568
	Add("backbone.bottom_up.res4.4.conv3.norm.weight", 1024); // 29798144
	Add("backbone.bottom_up.res4.4.conv3.norm.bias", 1024); // 29802240
	Add("backbone.bottom_up.res4.4.conv3.norm.running_mean", 1024); // 29806336
	Add("backbone.bottom_up.res4.4.conv3.norm.running_var", 1024); // 29810432
	Add("backbone.bottom_up.res4.5.conv1.weight", 262144); // 29814528
	Add("backbone.bottom_up.res4.5.conv1.norm.weight", 256); // 30863104
	Add("backbone.bottom_up.res4.5.conv1.norm.bias", 256); // 30864128
	Add("backbone.bottom_up.res4.5.conv1.norm.running_mean", 256); // 30865152
	Add("backbone.bottom_up.res4.5.conv1.norm.running_var", 256); // 30866176
	Add("backbone.bottom_up.res4.5.conv2.weight", 589824); // 30867200
	Add("backbone.bottom_up.res4.5.conv2.norm.weight", 256); // 33226496
	Add("backbone.bottom_up.res4.5.conv2.norm.bias", 256); // 33227520
	Add("backbone.bottom_up.res4.5.conv2.norm.running_mean", 256); // 33228544
	Add("backbone.bottom_up.res4.5.conv2.norm.running_var", 256); // 33229568
	Add("backbone.bottom_up.res4.5.conv3.weight", 262144); // 33230592
	Add("backbone.bottom_up.res4.5.conv3.norm.weight", 1024); // 34279168
	Add("backbone.bottom_up.res4.5.conv3.norm.bias", 1024); // 34283264
	Add("backbone.bottom_up.res4.5.conv3.norm.running_mean", 1024); // 34287360
	Add("backbone.bottom_up.res4.5.conv3.norm.running_var", 1024); // 34291456
	Add("backbone.bottom_up.res4.6.conv1.weight", 262144); // 34295552
	Add("backbone.bottom_up.res4.6.conv1.norm.weight", 256); // 35344128
	Add("backbone.bottom_up.res4.6.conv1.norm.bias", 256); // 35345152
	Add("backbone.bottom_up.res4.6.conv1.norm.running_mean", 256); // 35346176
	Add("backbone.bottom_up.res4.6.conv1.norm.running_var", 256); // 35347200
	Add("backbone.bottom_up.res4.6.conv2.weight", 589824); // 35348224
	Add("backbone.bottom_up.res4.6.conv2.norm.weight", 256); // 37707520
	Add("backbone.bottom_up.res4.6.conv2.norm.bias", 256); // 37708544
	Add("backbone.bottom_up.res4.6.conv2.norm.running_mean", 256); // 37709568
	Add("backbone.bottom_up.res4.6.conv2.norm.running_var", 256); // 37710592
	Add("backbone.bottom_up.res4.6.conv3.weight", 262144); // 37711616
	Add("backbone.bottom_up.res4.6.conv3.norm.weight", 1024); // 38760192
	Add("backbone.bottom_up.res4.6.conv3.norm.bias", 1024); // 38764288
	Add("backbone.bottom_up.res4.6.conv3.norm.running_mean", 1024); // 38768384
	Add("backbone.bottom_up.res4.6.conv3.norm.running_var", 1024); // 38772480
	Add("backbone.bottom_up.res4.7.conv1.weight", 262144); // 38776576
	Add("backbone.bottom_up.res4.7.conv1.norm.weight", 256); // 39825152
	Add("backbone.bottom_up.res4.7.conv1.norm.bias", 256); // 39826176
	Add("backbone.bottom_up.res4.7.conv1.norm.running_mean", 256); // 39827200
	Add("backbone.bottom_up.res4.7.conv1.norm.running_var", 256); // 39828224
	Add("backbone.bottom_up.res4.7.conv2.weight", 589824); // 39829248
	Add("backbone.bottom_up.res4.7.conv2.norm.weight", 256); // 42188544
	Add("backbone.bottom_up.res4.7.conv2.norm.bias", 256); // 42189568
	Add("backbone.bottom_up.res4.7.conv2.norm.running_mean", 256); // 42190592
	Add("backbone.bottom_up.res4.7.conv2.norm.running_var", 256); // 42191616
	Add("backbone.bottom_up.res4.7.conv3.weight", 262144); // 42192640
	Add("backbone.bottom_up.res4.7.conv3.norm.weight", 1024); // 43241216
	Add("backbone.bottom_up.res4.7.conv3.norm.bias", 1024); // 43245312
	Add("backbone.bottom_up.res4.7.conv3.norm.running_mean", 1024); // 43249408
	Add("backbone.bottom_up.res4.7.conv3.norm.running_var", 1024); // 43253504
	Add("backbone.bottom_up.res4.8.conv1.weight", 262144); // 43257600
	Add("backbone.bottom_up.res4.8.conv1.norm.weight", 256); // 44306176
	Add("backbone.bottom_up.res4.8.conv1.norm.bias", 256); // 44307200
	Add("backbone.bottom_up.res4.8.conv1.norm.running_mean", 256); // 44308224
	Add("backbone.bottom_up.res4.8.conv1.norm.running_var", 256); // 44309248
	Add("backbone.bottom_up.res4.8.conv2.weight", 589824); // 44310272
	Add("backbone.bottom_up.res4.8.conv2.norm.weight", 256); // 46669568
	Add("backbone.bottom_up.res4.8.conv2.norm.bias", 256); // 46670592
	Add("backbone.bottom_up.res4.8.conv2.norm.running_mean", 256); // 46671616
	Add("backbone.bottom_up.res4.8.conv2.norm.running_var", 256); // 46672640
	Add("backbone.bottom_up.res4.8.conv3.weight", 262144); // 46673664
	Add("backbone.bottom_up.res4.8.conv3.norm.weight", 1024); // 47722240
	Add("backbone.bottom_up.res4.8.conv3.norm.bias", 1024); // 47726336
	Add("backbone.bottom_up.res4.8.conv3.norm.running_mean", 1024); // 47730432
	Add("backbone.bottom_up.res4.8.conv3.norm.running_var", 1024); // 47734528
	Add("backbone.bottom_up.res4.9.conv1.weight", 262144); // 47738624
	Add("backbone.bottom_up.res4.9.conv1.norm.weight", 256); // 48787200
	Add("backbone.bottom_up.res4.9.conv1.norm.bias", 256); // 48788224
	Add("backbone.bottom_up.res4.9.conv1.norm.running_mean", 256); // 48789248
	Add("backbone.bottom_up.res4.9.conv1.norm.running_var", 256); // 48790272
	Add("backbone.bottom_up.res4.9.conv2.weight", 589824); // 48791296
	Add("backbone.bottom_up.res4.9.conv2.norm.weight", 256); // 51150592
	Add("backbone.bottom_up.res4.9.conv2.norm.bias", 256); // 51151616
	Add("backbone.bottom_up.res4.9.conv2.norm.running_mean", 256); // 51152640
	Add("backbone.bottom_up.res4.9.conv2.norm.running_var", 256); // 51153664
	Add("backbone.bottom_up.res4.9.conv3.weight", 262144); // 51154688
	Add("backbone.bottom_up.res4.9.conv3.norm.weight", 1024); // 52203264
	Add("backbone.bottom_up.res4.9.conv3.norm.bias", 1024); // 52207360
	Add("backbone.bottom_up.res4.9.conv3.norm.running_mean", 1024); // 52211456
	Add("backbone.bottom_up.res4.9.conv3.norm.running_var", 1024); // 52215552
	Add("backbone.bottom_up.res4.10.conv1.weight", 262144); // 52219648
	Add("backbone.bottom_up.res4.10.conv1.norm.weight", 256); // 53268224
	Add("backbone.bottom_up.res4.10.conv1.norm.bias", 256); // 53269248
	Add("backbone.bottom_up.res4.10.conv1.norm.running_mean", 256); // 53270272
	Add("backbone.bottom_up.res4.10.conv1.norm.running_var", 256); // 53271296
	Add("backbone.bottom_up.res4.10.conv2.weight", 589824); // 53272320
	Add("backbone.bottom_up.res4.10.conv2.norm.weight", 256); // 55631616
	Add("backbone.bottom_up.res4.10.conv2.norm.bias", 256); // 55632640
	Add("backbone.bottom_up.res4.10.conv2.norm.running_mean", 256); // 55633664
	Add("backbone.bottom_up.res4.10.conv2.norm.running_var", 256); // 55634688
	Add("backbone.bottom_up.res4.10.conv3.weight", 262144); // 55635712
	Add("backbone.bottom_up.res4.10.conv3.norm.weight", 1024); // 56684288
	Add("backbone.bottom_up.res4.10.conv3.norm.bias", 1024); // 56688384
	Add("backbone.bottom_up.res4.10.conv3.norm.running_mean", 1024); // 56692480
	Add("backbone.bottom_up.res4.10.conv3.norm.running_var", 1024); // 56696576
	Add("backbone.bottom_up.res4.11.conv1.weight", 262144); // 56700672
	Add("backbone.bottom_up.res4.11.conv1.norm.weight", 256); // 57749248
	Add("backbone.bottom_up.res4.11.conv1.norm.bias", 256); // 57750272
	Add("backbone.bottom_up.res4.11.conv1.norm.running_mean", 256); // 57751296
	Add("backbone.bottom_up.res4.11.conv1.norm.running_var", 256); // 57752320
	Add("backbone.bottom_up.res4.11.conv2.weight", 589824); // 57753344
	Add("backbone.bottom_up.res4.11.conv2.norm.weight", 256); // 60112640
	Add("backbone.bottom_up.res4.11.conv2.norm.bias", 256); // 60113664
	Add("backbone.bottom_up.res4.11.conv2.norm.running_mean", 256); // 60114688
	Add("backbone.bottom_up.res4.11.conv2.norm.running_var", 256); // 60115712
	Add("backbone.bottom_up.res4.11.conv3.weight", 262144); // 60116736
	Add("backbone.bottom_up.res4.11.conv3.norm.weight", 1024); // 61165312
	Add("backbone.bottom_up.res4.11.conv3.norm.bias", 1024); // 61169408
	Add("backbone.bottom_up.res4.11.conv3.norm.running_mean", 1024); // 61173504
	Add("backbone.bottom_up.res4.11.conv3.norm.running_var", 1024); // 61177600
	Add("backbone.bottom_up.res4.12.conv1.weight", 262144); // 61181696
	Add("backbone.bottom_up.res4.12.conv1.norm.weight", 256); // 62230272
	Add("backbone.bottom_up.res4.12.conv1.norm.bias", 256); // 62231296
	Add("backbone.bottom_up.res4.12.conv1.norm.running_mean", 256); // 62232320
	Add("backbone.bottom_up.res4.12.conv1.norm.running_var", 256); // 62233344
	Add("backbone.bottom_up.res4.12.conv2.weight", 589824); // 62234368
	Add("backbone.bottom_up.res4.12.conv2.norm.weight", 256); // 64593664
	Add("backbone.bottom_up.res4.12.conv2.norm.bias", 256); // 64594688
	Add("backbone.bottom_up.res4.12.conv2.norm.running_mean", 256); // 64595712
	Add("backbone.bottom_up.res4.12.conv2.norm.running_var", 256); // 64596736
	Add("backbone.bottom_up.res4.12.conv3.weight", 262144); // 64597760
	Add("backbone.bottom_up.res4.12.conv3.norm.weight", 1024); // 65646336
	Add("backbone.bottom_up.res4.12.conv3.norm.bias", 1024); // 65650432
	Add("backbone.bottom_up.res4.12.conv3.norm.running_mean", 1024); // 65654528
	Add("backbone.bottom_up.res4.12.conv3.norm.running_var", 1024); // 65658624
	Add("backbone.bottom_up.res4.13.conv1.weight", 262144); // 65662720
	Add("backbone.bottom_up.res4.13.conv1.norm.weight", 256); // 66711296
	Add("backbone.bottom_up.res4.13.conv1.norm.bias", 256); // 66712320
	Add("backbone.bottom_up.res4.13.conv1.norm.running_mean", 256); // 66713344
	Add("backbone.bottom_up.res4.13.conv1.norm.running_var", 256); // 66714368
	Add("backbone.bottom_up.res4.13.conv2.weight", 589824); // 66715392
	Add("backbone.bottom_up.res4.13.conv2.norm.weight", 256); // 69074688
	Add("backbone.bottom_up.res4.13.conv2.norm.bias", 256); // 69075712
	Add("backbone.bottom_up.res4.13.conv2.norm.running_mean", 256); // 69076736
	Add("backbone.bottom_up.res4.13.conv2.norm.running_var", 256); // 69077760
	Add("backbone.bottom_up.res4.13.conv3.weight", 262144); // 69078784
	Add("backbone.bottom_up.res4.13.conv3.norm.weight", 1024); // 70127360
	Add("backbone.bottom_up.res4.13.conv3.norm.bias", 1024); // 70131456
	Add("backbone.bottom_up.res4.13.conv3.norm.running_mean", 1024); // 70135552
	Add("backbone.bottom_up.res4.13.conv3.norm.running_var", 1024); // 70139648
	Add("backbone.bottom_up.res4.14.conv1.weight", 262144); // 70143744
	Add("backbone.bottom_up.res4.14.conv1.norm.weight", 256); // 71192320
	Add("backbone.bottom_up.res4.14.conv1.norm.bias", 256); // 71193344
	Add("backbone.bottom_up.res4.14.conv1.norm.running_mean", 256); // 71194368
	Add("backbone.bottom_up.res4.14.conv1.norm.running_var", 256); // 71195392
	Add("backbone.bottom_up.res4.14.conv2.weight", 589824); // 71196416
	Add("backbone.bottom_up.res4.14.conv2.norm.weight", 256); // 73555712
	Add("backbone.bottom_up.res4.14.conv2.norm.bias", 256); // 73556736
	Add("backbone.bottom_up.res4.14.conv2.norm.running_mean", 256); // 73557760
	Add("backbone.bottom_up.res4.14.conv2.norm.running_var", 256); // 73558784
	Add("backbone.bottom_up.res4.14.conv3.weight", 262144); // 73559808
	Add("backbone.bottom_up.res4.14.conv3.norm.weight", 1024); // 74608384
	Add("backbone.bottom_up.res4.14.conv3.norm.bias", 1024); // 74612480
	Add("backbone.bottom_up.res4.14.conv3.norm.running_mean", 1024); // 74616576
	Add("backbone.bottom_up.res4.14.conv3.norm.running_var", 1024); // 74620672
	Add("backbone.bottom_up.res4.15.conv1.weight", 262144); // 74624768
	Add("backbone.bottom_up.res4.15.conv1.norm.weight", 256); // 75673344
	Add("backbone.bottom_up.res4.15.conv1.norm.bias", 256); // 75674368
	Add("backbone.bottom_up.res4.15.conv1.norm.running_mean", 256); // 75675392
	Add("backbone.bottom_up.res4.15.conv1.norm.running_var", 256); // 75676416
	Add("backbone.bottom_up.res4.15.conv2.weight", 589824); // 75677440
	Add("backbone.bottom_up.res4.15.conv2.norm.weight", 256); // 78036736
	Add("backbone.bottom_up.res4.15.conv2.norm.bias", 256); // 78037760
	Add("backbone.bottom_up.res4.15.conv2.norm.running_mean", 256); // 78038784
	Add("backbone.bottom_up.res4.15.conv2.norm.running_var", 256); // 78039808
	Add("backbone.bottom_up.res4.15.conv3.weight", 262144); // 78040832
	Add("backbone.bottom_up.res4.15.conv3.norm.weight", 1024); // 79089408
	Add("backbone.bottom_up.res4.15.conv3.norm.bias", 1024); // 79093504
	Add("backbone.bottom_up.res4.15.conv3.norm.running_mean", 1024); // 79097600
	Add("backbone.bottom_up.res4.15.conv3.norm.running_var", 1024); // 79101696
	Add("backbone.bottom_up.res4.16.conv1.weight", 262144); // 79105792
	Add("backbone.bottom_up.res4.16.conv1.norm.weight", 256); // 80154368
	Add("backbone.bottom_up.res4.16.conv1.norm.bias", 256); // 80155392
	Add("backbone.bottom_up.res4.16.conv1.norm.running_mean", 256); // 80156416
	Add("backbone.bottom_up.res4.16.conv1.norm.running_var", 256); // 80157440
	Add("backbone.bottom_up.res4.16.conv2.weight", 589824); // 80158464
	Add("backbone.bottom_up.res4.16.conv2.norm.weight", 256); // 82517760
	Add("backbone.bottom_up.res4.16.conv2.norm.bias", 256); // 82518784
	Add("backbone.bottom_up.res4.16.conv2.norm.running_mean", 256); // 82519808
	Add("backbone.bottom_up.res4.16.conv2.norm.running_var", 256); // 82520832
	Add("backbone.bottom_up.res4.16.conv3.weight", 262144); // 82521856
	Add("backbone.bottom_up.res4.16.conv3.norm.weight", 1024); // 83570432
	Add("backbone.bottom_up.res4.16.conv3.norm.bias", 1024); // 83574528
	Add("backbone.bottom_up.res4.16.conv3.norm.running_mean", 1024); // 83578624
	Add("backbone.bottom_up.res4.16.conv3.norm.running_var", 1024); // 83582720
	Add("backbone.bottom_up.res4.17.conv1.weight", 262144); // 83586816
	Add("backbone.bottom_up.res4.17.conv1.norm.weight", 256); // 84635392
	Add("backbone.bottom_up.res4.17.conv1.norm.bias", 256); // 84636416
	Add("backbone.bottom_up.res4.17.conv1.norm.running_mean", 256); // 84637440
	Add("backbone.bottom_up.res4.17.conv1.norm.running_var", 256); // 84638464
	Add("backbone.bottom_up.res4.17.conv2.weight", 589824); // 84639488
	Add("backbone.bottom_up.res4.17.conv2.norm.weight", 256); // 86998784
	Add("backbone.bottom_up.res4.17.conv2.norm.bias", 256); // 86999808
	Add("backbone.bottom_up.res4.17.conv2.norm.running_mean", 256); // 87000832
	Add("backbone.bottom_up.res4.17.conv2.norm.running_var", 256); // 87001856
	Add("backbone.bottom_up.res4.17.conv3.weight", 262144); // 87002880
	Add("backbone.bottom_up.res4.17.conv3.norm.weight", 1024); // 88051456
	Add("backbone.bottom_up.res4.17.conv3.norm.bias", 1024); // 88055552
	Add("backbone.bottom_up.res4.17.conv3.norm.running_mean", 1024); // 88059648
	Add("backbone.bottom_up.res4.17.conv3.norm.running_var", 1024); // 88063744
	Add("backbone.bottom_up.res4.18.conv1.weight", 262144); // 88067840
	Add("backbone.bottom_up.res4.18.conv1.norm.weight", 256); // 89116416
	Add("backbone.bottom_up.res4.18.conv1.norm.bias", 256); // 89117440
	Add("backbone.bottom_up.res4.18.conv1.norm.running_mean", 256); // 89118464
	Add("backbone.bottom_up.res4.18.conv1.norm.running_var", 256); // 89119488
	Add("backbone.bottom_up.res4.18.conv2.weight", 589824); // 89120512
	Add("backbone.bottom_up.res4.18.conv2.norm.weight", 256); // 91479808
	Add("backbone.bottom_up.res4.18.conv2.norm.bias", 256); // 91480832
	Add("backbone.bottom_up.res4.18.conv2.norm.running_mean", 256); // 91481856
	Add("backbone.bottom_up.res4.18.conv2.norm.running_var", 256); // 91482880
	Add("backbone.bottom_up.res4.18.conv3.weight", 262144); // 91483904
	Add("backbone.bottom_up.res4.18.conv3.norm.weight", 1024); // 92532480
	Add("backbone.bottom_up.res4.18.conv3.norm.bias", 1024); // 92536576
	Add("backbone.bottom_up.res4.18.conv3.norm.running_mean", 1024); // 92540672
	Add("backbone.bottom_up.res4.18.conv3.norm.running_var", 1024); // 92544768
	Add("backbone.bottom_up.res4.19.conv1.weight", 262144); // 92548864
	Add("backbone.bottom_up.res4.19.conv1.norm.weight", 256); // 93597440
	Add("backbone.bottom_up.res4.19.conv1.norm.bias", 256); // 93598464
	Add("backbone.bottom_up.res4.19.conv1.norm.running_mean", 256); // 93599488
	Add("backbone.bottom_up.res4.19.conv1.norm.running_var", 256); // 93600512
	Add("backbone.bottom_up.res4.19.conv2.weight", 589824); // 93601536
	Add("backbone.bottom_up.res4.19.conv2.norm.weight", 256); // 95960832
	Add("backbone.bottom_up.res4.19.conv2.norm.bias", 256); // 95961856
	Add("backbone.bottom_up.res4.19.conv2.norm.running_mean", 256); // 95962880
	Add("backbone.bottom_up.res4.19.conv2.norm.running_var", 256); // 95963904
	Add("backbone.bottom_up.res4.19.conv3.weight", 262144); // 95964928
	Add("backbone.bottom_up.res4.19.conv3.norm.weight", 1024); // 97013504
	Add("backbone.bottom_up.res4.19.conv3.norm.bias", 1024); // 97017600
	Add("backbone.bottom_up.res4.19.conv3.norm.running_mean", 1024); // 97021696
	Add("backbone.bottom_up.res4.19.conv3.norm.running_var", 1024); // 97025792
	Add("backbone.bottom_up.res4.20.conv1.weight", 262144); // 97029888
	Add("backbone.bottom_up.res4.20.conv1.norm.weight", 256); // 98078464
	Add("backbone.bottom_up.res4.20.conv1.norm.bias", 256); // 98079488
	Add("backbone.bottom_up.res4.20.conv1.norm.running_mean", 256); // 98080512
	Add("backbone.bottom_up.res4.20.conv1.norm.running_var", 256); // 98081536
	Add("backbone.bottom_up.res4.20.conv2.weight", 589824); // 98082560
	Add("backbone.bottom_up.res4.20.conv2.norm.weight", 256); // 100441856
	Add("backbone.bottom_up.res4.20.conv2.norm.bias", 256); // 100442880
	Add("backbone.bottom_up.res4.20.conv2.norm.running_mean", 256); // 100443904
	Add("backbone.bottom_up.res4.20.conv2.norm.running_var", 256); // 100444928
	Add("backbone.bottom_up.res4.20.conv3.weight", 262144); // 100445952
	Add("backbone.bottom_up.res4.20.conv3.norm.weight", 1024); // 101494528
	Add("backbone.bottom_up.res4.20.conv3.norm.bias", 1024); // 101498624
	Add("backbone.bottom_up.res4.20.conv3.norm.running_mean", 1024); // 101502720
	Add("backbone.bottom_up.res4.20.conv3.norm.running_var", 1024); // 101506816
	Add("backbone.bottom_up.res4.21.conv1.weight", 262144); // 101510912
	Add("backbone.bottom_up.res4.21.conv1.norm.weight", 256); // 102559488
	Add("backbone.bottom_up.res4.21.conv1.norm.bias", 256); // 102560512
	Add("backbone.bottom_up.res4.21.conv1.norm.running_mean", 256); // 102561536
	Add("backbone.bottom_up.res4.21.conv1.norm.running_var", 256); // 102562560
	Add("backbone.bottom_up.res4.21.conv2.weight", 589824); // 102563584
	Add("backbone.bottom_up.res4.21.conv2.norm.weight", 256); // 104922880
	Add("backbone.bottom_up.res4.21.conv2.norm.bias", 256); // 104923904
	Add("backbone.bottom_up.res4.21.conv2.norm.running_mean", 256); // 104924928
	Add("backbone.bottom_up.res4.21.conv2.norm.running_var", 256); // 104925952
	Add("backbone.bottom_up.res4.21.conv3.weight", 262144); // 104926976
	Add("backbone.bottom_up.res4.21.conv3.norm.weight", 1024); // 105975552
	Add("backbone.bottom_up.res4.21.conv3.norm.bias", 1024); // 105979648
	Add("backbone.bottom_up.res4.21.conv3.norm.running_mean", 1024); // 105983744
	Add("backbone.bottom_up.res4.21.conv3.norm.running_var", 1024); // 105987840
	Add("backbone.bottom_up.res4.22.conv1.weight", 262144); // 105991936
	Add("backbone.bottom_up.res4.22.conv1.norm.weight", 256); // 107040512
	Add("backbone.bottom_up.res4.22.conv1.norm.bias", 256); // 107041536
	Add("backbone.bottom_up.res4.22.conv1.norm.running_mean", 256); // 107042560
	Add("backbone.bottom_up.res4.22.conv1.norm.running_var", 256); // 107043584
	Add("backbone.bottom_up.res4.22.conv2.weight", 589824); // 107044608
	Add("backbone.bottom_up.res4.22.conv2.norm.weight", 256); // 109403904
	Add("backbone.bottom_up.res4.22.conv2.norm.bias", 256); // 109404928
	Add("backbone.bottom_up.res4.22.conv2.norm.running_mean", 256); // 109405952
	Add("backbone.bottom_up.res4.22.conv2.norm.running_var", 256); // 109406976
	Add("backbone.bottom_up.res4.22.conv3.weight", 262144); // 109408000
	Add("backbone.bottom_up.res4.22.conv3.norm.weight", 1024); // 110456576
	Add("backbone.bottom_up.res4.22.conv3.norm.bias", 1024); // 110460672
	Add("backbone.bottom_up.res4.22.conv3.norm.running_mean", 1024); // 110464768
	Add("backbone.bottom_up.res4.22.conv3.norm.running_var", 1024); // 110468864
	Add("backbone.bottom_up.res5.0.shortcut.weight", 2097152); // 110472960
	Add("backbone.bottom_up.res5.0.shortcut.norm.weight", 2048); // 118861568
	Add("backbone.bottom_up.res5.0.shortcut.norm.bias", 2048); // 118869760
	Add("backbone.bottom_up.res5.0.shortcut.norm.running_mean", 2048); // 118877952
	Add("backbone.bottom_up.res5.0.shortcut.norm.running_var", 2048); // 118886144
	Add("backbone.bottom_up.res5.0.conv1.weight", 524288); // 118894336
	Add("backbone.bottom_up.res5.0.conv1.norm.weight", 512); // 120991488
	Add("backbone.bottom_up.res5.0.conv1.norm.bias", 512); // 120993536
	Add("backbone.bottom_up.res5.0.conv1.norm.running_mean", 512); // 120995584
	Add("backbone.bottom_up.res5.0.conv1.norm.running_var", 512); // 120997632
	Add("backbone.bottom_up.res5.0.conv2.weight", 2359296); // 120999680
	Add("backbone.bottom_up.res5.0.conv2.norm.weight", 512); // 130436864
	Add("backbone.bottom_up.res5.0.conv2.norm.bias", 512); // 130438912
	Add("backbone.bottom_up.res5.0.conv2.norm.running_mean", 512); // 130440960
	Add("backbone.bottom_up.res5.0.conv2.norm.running_var", 512); // 130443008
	Add("backbone.bottom_up.res5.0.conv3.weight", 1048576); // 130445056
	Add("backbone.bottom_up.res5.0.conv3.norm.weight", 2048); // 134639360
	Add("backbone.bottom_up.res5.0.conv3.norm.bias", 2048); // 134647552
	Add("backbone.bottom_up.res5.0.conv3.norm.running_mean", 2048); // 134655744
	Add("backbone.bottom_up.res5.0.conv3.norm.running_var", 2048); // 134663936
	Add("backbone.bottom_up.res5.1.conv1.weight", 1048576); // 134672128
	Add("backbone.bottom_up.res5.1.conv1.norm.weight", 512); // 138866432
	Add("backbone.bottom_up.res5.1.conv1.norm.bias", 512); // 138868480
	Add("backbone.bottom_up.res5.1.conv1.norm.running_mean", 512); // 138870528
	Add("backbone.bottom_up.res5.1.conv1.norm.running_var", 512); // 138872576
	Add("backbone.bottom_up.res5.1.conv2.weight", 2359296); // 138874624
	Add("backbone.bottom_up.res5.1.conv2.norm.weight", 512); // 148311808
	Add("backbone.bottom_up.res5.1.conv2.norm.bias", 512); // 148313856
	Add("backbone.bottom_up.res5.1.conv2.norm.running_mean", 512); // 148315904
	Add("backbone.bottom_up.res5.1.conv2.norm.running_var", 512); // 148317952
	Add("backbone.bottom_up.res5.1.conv3.weight", 1048576); // 148320000
	Add("backbone.bottom_up.res5.1.conv3.norm.weight", 2048); // 152514304
	Add("backbone.bottom_up.res5.1.conv3.norm.bias", 2048); // 152522496
	Add("backbone.bottom_up.res5.1.conv3.norm.running_mean", 2048); // 152530688
	Add("backbone.bottom_up.res5.1.conv3.norm.running_var", 2048); // 152538880
	Add("backbone.bottom_up.res5.2.conv1.weight", 1048576); // 152547072
	Add("backbone.bottom_up.res5.2.conv1.norm.weight", 512); // 156741376
	Add("backbone.bottom_up.res5.2.conv1.norm.bias", 512); // 156743424
	Add("backbone.bottom_up.res5.2.conv1.norm.running_mean", 512); // 156745472
	Add("backbone.bottom_up.res5.2.conv1.norm.running_var", 512); // 156747520
	Add("backbone.bottom_up.res5.2.conv2.weight", 2359296); // 156749568
	Add("backbone.bottom_up.res5.2.conv2.norm.weight", 512); // 166186752
	Add("backbone.bottom_up.res5.2.conv2.norm.bias", 512); // 166188800
	Add("backbone.bottom_up.res5.2.conv2.norm.running_mean", 512); // 166190848
	Add("backbone.bottom_up.res5.2.conv2.norm.running_var", 512); // 166192896
	Add("backbone.bottom_up.res5.2.conv3.weight", 1048576); // 166194944
	Add("backbone.bottom_up.res5.2.conv3.norm.weight", 2048); // 170389248
	Add("backbone.bottom_up.res5.2.conv3.norm.bias", 2048); // 170397440
	Add("backbone.bottom_up.res5.2.conv3.norm.running_mean", 2048); // 170405632
	Add("backbone.bottom_up.res5.2.conv3.norm.running_var", 2048); // 170413824

	return DataDir() + "\\model_final_f6e8b1.data";
}
