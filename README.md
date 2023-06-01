# BOD

This is the official PyTorch implementation for the paper:
> Zongwei Wang, Min Gao*, Wentao Li*, Junliang Yu, Linxin Guo, Hongzhi Yin. Efficient Bi-Level Optimization for Recommendation Denoising. KDD 2023.

<h2>Requirements</h2>
	
```
numba==0.53.1
numpy==1.20.3
scipy==1.6.2
torch>=1.7.0
```

<h2>Usage</h2>
<ol>
<li>Configure the xx.conf file in the directory named conf. (xx is the name of the model you want to run)</li>
<li>Run main.py and choose the model you want to run.</li>
</ol>

<h2>Acknowledgement</h2>

The implementation is based on the open-source recommendation library [SelfRec](https://github.com/Coder-Yu/SELFRec).

Please cite the following papers as the references if you use our codes.

```bibtex
@article{yu2022self,
  title={Self-supervised learning for recommender systems: A survey},
  author={Yu, Junliang and Yin, Hongzhi and Xia, Xin and Chen, Tong and Li, Jundong and Huang, Zi},
  journal={arXiv preprint arXiv:2203.15876},
  year={2022}
}

@inproceedings{wang2023efficient,
  author = {Zongwei Wang, Min Gao, Wentao Li, Junliang Yu, Linxin Guo, Hongzhi Yin.},
  title = {Efficient Bi-Level Optimization for Recommendation Denoising},
  booktitle = {{KDD}},
  year = {2023}
}
