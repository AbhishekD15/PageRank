# PageRank
Pagerank given node and edges.

### Pagerank implementation in Python

Sample input file: http://snap.stanford.edu/data/web-Google.html

option:

```bash
-f --file | Target file. 
```
Format: 
  # Comment
  From_node_id(tab)To_node_id
  From_node_id(tab)To_node_id
  From_node_id(tab)To_node_id
  
```bash
-m --matrix | Use tradition matrix mode instead of Graph mode,requires more memory
```

```bash
-p --probability | Probability for jumping to other random webpage. Solves dead-ends and spider traps
```


```bash
-t --processes | Multi-processes for large matrix
```
