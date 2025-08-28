# Author: Zeheng Bai
##### TRAINING DNABERT WITHOUT PRETRAINING #####
# Device: GPU #
from basicsetting import *
from readfasta import *
from INHERITModels import *
from Dataset_config import *
from IHT_config_gvd_lstmv1_new import *
import math
import matplotlib.pyplot as plt
from sophia import SophiaG

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description=__doc__)
    PARSER.add_argument("--bertdir",
                        type=str,
                        default='')
    PARSER.add_argument("--checkpoint",
                        type=str,
                        default = '')
    PARSER.add_argument("--outdir",
                        type=str,
                        required=True)

    args = PARSER.parse_args()

    ##### Hyperparamters: in Network_config #####
    pid = 'checkpoints'
    path = os.getcwd() + "/" + str(pid)
    if os.path.exists(path) is False:
        os.makedirs(path)
    config = BertConfig.from_pretrained(CONFIG_PATH)
    early_stopping = EarlyStopping(checkpoint = pid + '/' + 'bestvalloss_' + args.outdir, patience=5, verbose=True)
    tokenizer = DNATokenizer.from_pretrained(CONFIG_PATH)
    X_bac_tr = read_fasta(open(BAC_TR_PATH), KMERS, SEGMENT_LENGTH)
    X_pha_tr = read_fasta(open(PHA_TR_PATH), KMERS, SEGMENT_LENGTH)
    tokenizer.cls_token = '[BAC]'
    bac_baclab_examples = tokenizer.batch_encode_plus(X_bac_tr, add_special_tokens=True)["input_ids"]
    pha_baclab_examples = tokenizer.batch_encode_plus(X_pha_tr, add_special_tokens=True)["input_ids"]
    tokenizer.cls_token = '[PHA]'
    bac_phalab_examples = tokenizer.batch_encode_plus(X_bac_tr, add_special_tokens=True)["input_ids"]
    pha_phalab_examples = tokenizer.batch_encode_plus(X_pha_tr, add_special_tokens=True)["input_ids"]
    X_train = bac_baclab_examples + pha_phalab_examples + pha_baclab_examples + bac_phalab_examples
    y_train = torch.cat((torch.zeros(len(bac_baclab_examples)), torch.ones(len(pha_phalab_examples)), torch.tensor([2]*(len(pha_baclab_examples)+ len(bac_phalab_examples))))) 
    y_train = y_train.to(torch.long).unsqueeze(1)
    X_bac_val = read_fasta(open(BAC_VAL_PATH), KMERS, SEGMENT_LENGTH)
    X_pha_val = read_fasta(open(PHA_VAL_PATH), KMERS, SEGMENT_LENGTH)
    tokenizer.cls_token = '[BAC]'
    bac_baclab_val_examples = tokenizer.batch_encode_plus(X_bac_val, add_special_tokens=True)["input_ids"]
    pha_baclab_val_examples = tokenizer.batch_encode_plus(X_pha_val, add_special_tokens=True)["input_ids"]
    tokenizer.cls_token = '[PHA]'
    bac_phalab_val_examples = tokenizer.batch_encode_plus(X_bac_val, add_special_tokens=True)["input_ids"]
    pha_phalab_val_examples = tokenizer.batch_encode_plus(X_pha_val, add_special_tokens=True)["input_ids"]
    X_val = bac_baclab_val_examples + pha_phalab_val_examples + pha_baclab_val_examples + bac_phalab_val_examples
    y_val = torch.cat((torch.zeros(len(bac_baclab_val_examples)), torch.ones(len(pha_phalab_val_examples)), torch.tensor([2]*(len(pha_baclab_val_examples)+ len(bac_phalab_val_examples))))) 
    y_val = y_val.to(torch.long).unsqueeze(1)
    train_data = cond_IHTDataset(x=torch.tensor(X_train), y=y_train)
    train_loader = DataLoader(train_data, batch_size=TR_BATCHSIZE, shuffle=True, num_workers=TR_WORKERS)
    val_data = cond_IHTDataset(x=torch.tensor(X_val), y=y_val)
    val_loader = DataLoader(val_data, batch_size=VAL_BATCHSIZE, shuffle=True, num_workers=VAL_WORKERS)
    bertmodel = Baseline_conditional_BERT(freeze_bert=False, config=config, bert_dir = args.bertdir)
    bert_params = list(map(id, bertmodel.bert.parameters()))
    new_params = filter(lambda p: id(p) not in bert_params, bertmodel.parameters())
    opt = torch.optim.Adam([{'params': bertmodel.bert.parameters(), 'lr': LEARNING_RATE},
                                  {'params': new_params}], lr=LEARNING_RATE)
    if torch.cuda.device_count() > 1:
        bertmodel = torch.nn.DataParallel(bertmodel)
    if args.checkpoint != '':
        sdict = torch.load(args.checkpoint)
        bertmodel.load_state_dict(sdict)
    bertmodel.to(device)
    sigmoid = torch.nn.Sigmoid()
    loss_func = torch.nn.CrossEntropyLoss()
    print("The checkpoints will be in: " + path)
    print("start to train")
    train_loss_value=[]
    train_acc_value=[]
    val_loss_value=[]
    val_acc_value=[]
    for epoch in range(EPOCHS):
        running_loss = 0.0
        valrun_loss = 0.0
        sum_correct = 0
        sum_total = 0
        t = len(train_loader.dataset)
        bertmodel.train()
        with tqdm(total=100) as pbar:
            batchsize = TR_BATCHSIZE
            for i, (x, y) in enumerate(train_loader):
                #torch.save(x, 'x_sample.pt')
                X_seq = x
                X_seq = X_seq.to(device)
                #torch.save(X_seq, 'A.pt')
                Label = Variable(y)
                Label = Label.squeeze(0)
                Label = Label.squeeze(1).to(device)
                opt.zero_grad()
                out = bertmodel(input_ids=X_seq)
                #out = bertmodel(input_ids=X_seq['input_ids'].squeeze(1))
                loss = loss_func(out, Label.to(torch.long))
                loss.backward()
                opt.step()
                # print statistics
                running_loss += loss.item()
                o1 = torch.argmax(out.softmax(dim=-1), dim=1).unsqueeze(1)
                predicted = torch.round_(o1)
                sum_total += Label.size(0)
                sum_correct += (predicted == Label.unsqueeze(1)).sum().item()
                pbar.update(100 * batchsize / t)
            pbar.close()
        print("epochs={}, mean loss={}, accuracy={}"
            .format(epoch + 1, running_loss * batchsize / t, float(sum_correct / sum_total)))
        train_loss_value.append(running_loss * batchsize / t)
        train_acc_value.append(float(sum_correct / sum_total))
        bertmodel.eval()
        t = len(val_loader.dataset)
        batchsize = VAL_BATCHSIZE
        sum_correct = 0
        sum_total = 0
        for i, (x, y) in enumerate(val_loader):
            X_seq = x
            X_seq = X_seq.to(device)
            Label = Variable(y)
            Label = Label.squeeze(0)
            Label = Label.squeeze(1).to(device)
            val_output = bertmodel(input_ids=X_seq)
            #val_output = bertmodel(input_ids=X_seq['input_ids'].squeeze(1))
            loss = loss_func(val_output, Label.to(torch.long))
            valrun_loss += loss.item()
            val_o1 = torch.argmax(val_output.softmax(dim=-1), dim=1).unsqueeze(1)
            predicted = torch.round_(val_o1)
            sum_total += Label.size(0)
            sum_correct += (predicted == Label.unsqueeze(1)).sum().item()
        val_acc = float(sum_correct / sum_total)
        print("epochs={}, val loss={}, val accuracy={}"
            .format(epoch + 1, valrun_loss * batchsize / t, float(sum_correct / sum_total)))
        val_loss_value.append(valrun_loss * batchsize / t)
        val_acc_value.append(float(sum_correct / sum_total))
        plt.cla()
        plt.plot(range(1,len(train_loss_value) + 1), train_loss_value, marker='o', mec='r', mfc='w', label=u'Training Loss')
        plt.plot(range(1,len(train_loss_value) + 1), val_loss_value, marker='*', ms=10, label=u'Validation Loss')
        plt.legend()
        plt.xticks(range(1,len(train_loss_value) + 1), range(1,len(train_loss_value) + 1), rotation=45)
        plt.margins(0)
        plt.xlabel(u"Epochs")
        plt.ylabel("Loss")
        plt.title("Loss plot")
        plt.savefig("Lossplot_conditional_bert_base_lstmv1_new_new_0522_lr5e-6_gvd.png")
        plt.cla()
        plt.plot(range(1,len(train_acc_value) + 1), train_acc_value, marker='o', mec='r', mfc='w', label=u'Training Accuracy')
        plt.plot(range(1,len(train_acc_value) + 1), val_acc_value, marker='*', ms=10, label=u'Validation Accuracy')
        plt.legend()
        plt.ylim(0.4, 1)
        plt.xticks(range(1,len(train_acc_value) + 1), range(1,len(train_acc_value) + 1), rotation=45)
        plt.margins(0)
        plt.xlabel(u"Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy plot")
        plt.savefig("Accplot_conditional_bert_base_lstmv1_new_new_0522_lr5e-6_gvd.png")
        torch.save(bertmodel.state_dict(), pid + '/' + 'checkpoint_' + str(epoch) + '_' + args.outdir)
        early_stopping(valrun_loss * batchsize / t, bertmodel)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    torch.save(bertmodel.state_dict(), args.outdir)

