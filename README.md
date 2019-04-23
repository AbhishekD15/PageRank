# PageRank
Pagerank with given node and edges.
See: https://levyhsu.com/2019/04/page-rank-in-python/

### Pagerank implementation in Python

Sample input file: http://snap.stanford.edu/data/web-Google.html

Format: 
  '# Comment'<br/>
  From_node_id(tab)To_node_id<br/>
  From_node_id(tab)To_node_id<br/>
  From_node_id(tab)To_node_id<br/>


Option:

```bash
-f --file | Target file. 
```
  
```bash
-m --matrix | Use tradition matrix mode instead of Graph mode,requires more memory
```

```bash
-p --probability | Probability for jumping to other random webpage. Solves dead-ends and spider traps
```


```bash
-t --processes | Multi-processes for large matrix
```

Run:

```bash
python3 pagerank.py -f web-Google.txt
```
