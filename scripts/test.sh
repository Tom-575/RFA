###############################test demo
CUDA_VISIBLE_DEVICES=1 python test.py --batch_size 200 --test_num 10 \
--RFA_model_ckpt 'checkpoints/CIFAR10-RFA-FB-AF3-demo.pth' \
"Clean()" \
"PGDLinfAttack(model, eps=8/255, num_iters=100)" \
"PGDL2Attack(model, eps=1.0, num_iters=100)" \
"CWL2Attack(model, num_classes=10, num_iters=100)" \
"AALinfAttack(model, 'cifar10', bound=8/255)" \
"AAL2Attack(model, 'cifar10', bound=1.0)" \
"ReColorAttack(model, num_iters=100)" \
"STAAttack(model, num_iters=100)" \
"BIMLinfAttack(model, num_iters=100, eps=8./255, targeted=False)" \
"BIML2Attack(model, num_iters=100, eps=1.0, targeted=False)" \
"DFLinfAttack(model, num_classes=10, num_iters=100, eps=8./255, targeted=False)"

