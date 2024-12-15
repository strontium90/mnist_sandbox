
import torch


def train_loop(train_loader, net, criterion, optimizer, device,
               mbatch_loss_group=-1):
    net.train()
    running_loss = 0.0
    mbatch_losses = []
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # following condition False by default, unless mbatch_loss_group > 0
        if i % mbatch_loss_group == mbatch_loss_group - 1:
            mbatch_losses.append(running_loss / mbatch_loss_group)
            running_loss = 0.0
    if mbatch_loss_group > 0:
        return mbatch_losses


def validation_loop(val_loader, net, criterion, num_classes, device,
                    multi_task=False, th_multi_task=0.5, one_hot=False, class_metrics=False):
    net.eval()
    loss = 0
    correct = 0
    size = len(val_loader.dataset)
    class_total = {label:0 for label in range(num_classes)}
    class_tp = {label:0 for label in range(num_classes)}
    class_fp = {label:0 for label in range(num_classes)}
    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item() * images.size(0)
            if not multi_task:    
                predictions = torch.zeros_like(outputs)
                predictions[torch.arange(outputs.shape[0]), torch.argmax(outputs, dim=1)] = 1.0
            else:
                predictions = torch.where(outputs > th_multi_task, 1.0, 0.0)
            if not one_hot:
                labels_mat = torch.zeros_like(outputs)
                labels_mat[torch.arange(outputs.shape[0]), labels] = 1.0
                labels = labels_mat
                
            tps = predictions * labels
            fps = predictions - tps
            
            tps = tps.sum(dim=0)
            fps = fps.sum(dim=0)
            lbls = labels.sum(dim=0)  
                
            for c in range(num_classes):
                class_tp[c] += tps[c]
                class_fp[c] += fps[c]
                class_total[c] += lbls[c]
                    
            correct += tps.sum()

    class_prec = []
    class_recall = []
    freqs = []
    for c in range(num_classes):
        class_prec.append(0 if class_tp[c] == 0 else
                          class_tp[c] / (class_tp[c] + class_fp[c]))
        class_recall.append(0 if class_tp[c] == 0 else
                            class_tp[c] / class_total[c])
        freqs.append(class_total[c])

    freqs = torch.tensor(freqs)
    class_weights = 1. / freqs
    class_weights /= class_weights.sum()
    class_prec = torch.tensor(class_prec)
    class_recall = torch.tensor(class_recall)
    prec = (class_prec * class_weights).sum()
    recall = (class_recall * class_weights).sum()
    f1 = 2. / (1/prec + 1/recall)
    val_loss = loss / size
    accuracy = correct / size
    results = {"loss": val_loss, "accuracy": accuracy, "f1": f1,\
               "precision": prec, "recall": recall}

    if class_metrics:
        class_results = []
        for p, r in zip(class_prec, class_recall):
            f1 = (0 if p == r == 0 else 2. / (1/p + 1/r))
            class_results.append({"f1": f1, "precision": p, "recall": r})
        results = results, class_results

    return results


def update_graphs(summary_writer, epoch, train_results, test_results,
                  train_class_results=None, test_class_results=None,
                  mbatch_group=-1, mbatch_count=0, mbatch_losses=None):
    if mbatch_group > 0:
        for i in range(len(mbatch_losses)):
            summary_writer.add_scalar("Losses/Train mini-batches",
                                  mbatch_losses[i],
                                  epoch * mbatch_count + (i+1)*mbatch_group)

    summary_writer.add_scalars("Losses/Train Loss vs Test Loss",
                               {"Train Loss" : train_results["loss"],
                                "Test Loss" : test_results["loss"]},
                               (epoch + 1) if not mbatch_group > 0
                                     else (epoch + 1) * mbatch_count)

    summary_writer.add_scalars("Metrics/Train Accuracy vs Test Accuracy",
                               {"Train Accuracy" : train_results["accuracy"],
                                "Test Accuracy" : test_results["accuracy"]},
                               (epoch + 1) if not mbatch_group > 0
                                     else (epoch + 1) * mbatch_count)

    summary_writer.add_scalars("Metrics/Train F1 vs Test F1",
                               {"Train F1" : train_results["f1"],
                                "Test F1" : test_results["f1"]},
                               (epoch + 1) if not mbatch_group > 0
                                     else (epoch + 1) * mbatch_count)

    summary_writer.add_scalars("Metrics/Train Precision vs Test Precision",
                               {"Train Precision" : train_results["precision"],
                                "Test Precision" : test_results["precision"]},
                               (epoch + 1) if not mbatch_group > 0
                                     else (epoch + 1) * mbatch_count)

    summary_writer.add_scalars("Metrics/Train Recall vs Test Recall",
                               {"Train Recall" : train_results["recall"],
                                "Test Recall" : test_results["recall"]},
                               (epoch + 1) if not mbatch_group > 0
                                     else (epoch + 1) * mbatch_count)

    if train_class_results and test_class_results:
        for i in range(len(train_class_results)):
            summary_writer.add_scalars(f"Class Metrics/Class {i}/Train F1 vs Test F1",
                                       {"Train F1" : train_class_results[i]["f1"],
                                        "Test F1" : test_class_results[i]["f1"]},
                                       (epoch + 1) if not mbatch_group > 0
                                             else (epoch + 1) * mbatch_count)

            summary_writer.add_scalars(f"Class Metrics/Class {i}/Train Precision vs Test Precision",
                                       {"Train Precision" : train_class_results[i]["precision"],
                                        "Test Precision" : test_class_results[i]["precision"]},
                                       (epoch + 1) if not mbatch_group > 0
                                             else (epoch + 1) * mbatch_count)

            summary_writer.add_scalars(f"Class Metrics/Class {i}/Train Recall vs Test Recall",
                                       {"Train Recall" : train_class_results[i]["recall"],
                                        "Test Recall" : test_class_results[i]["recall"]},
                                       (epoch + 1) if not mbatch_group > 0
                                             else (epoch + 1) * mbatch_count)
    summary_writer.flush()
    

