#!/usr/bin/env python3
import csv, json, math
from pathlib import Path
import numpy as np

def safe_div(a,b):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan,dtype=float)
    return np.divide(a,b,out=out,where=b!=0)

def jf(x):
    x=float(x)
    return None if not math.isfinite(x) else x

def names_ok(names,n):
    names=[str(x) for x in names]
    if len(names)==n: return names
    if len(names)==n+1 and any(k in names[0].lower() for k in ("background","unlabeled","unlabelled","ignore")):
        return names[1:]
    raise ValueError(f"class_names={len(names)} pero hist={n}x{n}")

def write_matrix(path,matrix,names,dec=None):
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["GT\\Pred"]+list(names))
        for i,name in enumerate(names):
            vals=[]
            for v in matrix[i]:
                if dec is None: vals.append(int(round(float(v))))
                else: vals.append("" if not math.isfinite(float(v)) else f"{float(v):.{dec}f}")
            w.writerow([name]+vals)

def save_phase4_metrics(hist,class_names,output_dir,subset_name="test",existing_metrics=None):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    hist=np.asarray(hist,dtype=float)
    if hist.ndim!=2 or hist.shape[0]!=hist.shape[1]: raise ValueError("hist debe ser CxC")
    n=hist.shape[0]; names=names_ok(class_names,n)
    tp=np.diag(hist); gt=hist.sum(1); pred=hist.sum(0)
    fp=pred-tp; fn=gt-tp; tn=hist.sum()-(tp+fp+fn)
    precision=safe_div(tp,tp+fp)
    recall=safe_div(tp,tp+fn)
    f1=safe_div(2*precision*recall,precision+recall)
    iou=safe_div(tp,tp+fp+fn)
    specificity=safe_div(tn,tn+fp)
    row_norm=safe_div(hist,gt[:,None])
    col_norm=safe_div(hist,pred[None,:])

    rows=[]
    for i,name in enumerate(names):
        rows.append({
            "internal_id":i,"class":name,"gt_pixels":int(gt[i]),"pred_pixels":int(pred[i]),
            "TP":int(tp[i]),"FP":int(fp[i]),"FN":int(fn[i]),"TN":int(tn[i]),
            "gt_present":bool(gt[i]>0),"pred_present":bool(pred[i]>0),
            "precision":jf(precision[i]),"recall":jf(recall[i]),"f1":jf(f1[i]),
            "iou":jf(iou[i]),"specificity":jf(specificity[i])
        })
    fields=list(rows[0].keys()) if rows else []
    with (out/"per_class_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    write_matrix(out/"confusion_matrix_raw.csv",hist,names)
    write_matrix(out/"confusion_matrix_row_normalized.csv",row_norm,names,8)
    write_matrix(out/"confusion_matrix_column_normalized.csv",col_norm,names,8)
    np.save(out/"confusion_matrix.npy",hist)

    total=float(hist.sum())
    gt_present=gt>0
    pred_present=pred>0
    summary={
        "subset":subset_name,
        "num_classes":n,
        "total_evaluated_pixels":int(total),
        "classes_with_gt":int(gt_present.sum()),
        "classes_with_predictions":int(pred_present.sum()),
        "pixel_accuracy":jf(tp.sum()/total) if total else None,
        "macro_precision_defined":jf(np.nanmean(precision)) if np.any(np.isfinite(precision)) else None,
        "macro_recall_gt_present":jf(np.mean(recall[gt_present])) if np.any(gt_present) else None,
        "macro_f1_defined":jf(np.nanmean(f1)) if np.any(np.isfinite(f1)) else None,
        "mean_iou_gt_present":jf(np.mean(iou[gt_present])) if np.any(gt_present) else None,
        "existing_compute_score_metrics":existing_metrics or {},
        "per_class":rows,
        "definitions":{
            "precision":"TP/(TP+FP)",
            "recall":"TP/(TP+FN)",
            "f1":"2PR/(P+R)",
            "iou":"TP/(TP+FP+FN)",
            "row_normalized_confusion":"P(predicted | ground truth)"
        }
    }
    (out/"summary_metrics.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")

    def pct(x):
        return "N/A" if not math.isfinite(float(x)) else f"{100*float(x):.3f}%"
    lines=[f"SAR PHASE 4 METRICS — {subset_name}","="*92,
           f"Total evaluated pixels : {int(total)}",
           f"Classes with GT        : {int(gt_present.sum())}/{n}",
           f"Pixel accuracy         : {pct(tp.sum()/total) if total else 'N/A'}","",
           f"{'Class':24s} {'GT px':>11s} {'Pred px':>11s} {'Prec.':>9s} {'Recall':>9s} {'F1':>9s} {'IoU':>9s}",
           "-"*96]
    for i,name in enumerate(names):
        lines.append(f"{name[:24]:24s} {int(gt[i]):11d} {int(pred[i]):11d} {pct(precision[i]):>9s} {pct(recall[i]):>9s} {pct(f1[i]):>9s} {pct(iou[i]):>9s}")
    if existing_metrics:
        lines+=["","Repository compute_score():"]
        for k,v in existing_metrics.items(): lines.append(f"  {k:20s}: {v}")
    (out/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        m=np.nan_to_num(row_norm*100,nan=0.0)
        fig=plt.figure(figsize=(14,11)); ax=fig.add_subplot(111)
        im=ax.imshow(m,aspect="auto")
        ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
        ax.set_xticklabels(names,rotation=65,ha="right"); ax.set_yticklabels(names)
        ax.set_xlabel("Predicted class"); ax.set_ylabel("Ground-truth class")
        ax.set_title(f"Row-normalized confusion matrix (%) — {subset_name}")
        fig.colorbar(im,ax=ax,label="% of GT-class pixels")
        fig.tight_layout(); fig.savefig(out/"confusion_matrix_row_normalized.png",dpi=200,bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        (out/"heatmap_warning.txt").write_text(repr(e),encoding="utf-8")
    return summary
