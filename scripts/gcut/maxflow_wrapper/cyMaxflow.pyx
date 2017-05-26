# distutils: language = c++
import ctypes
import math
from libcpp.vector cimport vector
from libcpp cimport bool


cdef extern from "maxflow.h" namespace "maxflow":
	cdef cppclass Maxflow:
		Maxflow(int estimated_num_of_nodes, int estimated_num_of_edges, bool add_nodes_auto);

		void add_nodes(int num);
		void add_tweights(int node, float source_w, float sink_w);
		void add_multiple_tweights(vector[int] nodes, vector[float] source_ws, vector[float] sink_ws);
		void add_edge(int node_i, int node_j, float capacity, float rev_capacity);
		void add_multiple_edges(vector[int] node_is, vector[int] node_js, vector[float] capacities, vector[float] rev_capacities);

		vector[int] get_sink_nodes();

		float maxflow();

cdef class PyMaxflow:
	cdef Maxflow *thisptr

	def __cinit__(self, estimated_num_of_nodes, estimated_num_of_edges, add_nodes_auto):
		self.thisptr = new Maxflow(estimated_num_of_nodes, estimated_num_of_edges, add_nodes_auto)
	def __dealloc__(self):
		del self.thisptr
	def add_nodes(self, num=1):
		self.thisptr.add_nodes(num)
	def add_tweights(self, node, source_w, sink_w):
		self.thisptr.add_tweights(node, source_w, sink_w)
	def add_multiple_tweights(self, nodes, source_ws, sink_ws):
		self.thisptr.add_multiple_tweights(nodes, source_ws, sink_ws)
	def add_edge(self, node_i, node_j, capacity, rev_capacity):
		self.thisptr.add_edge(node_i, node_j, capacity, rev_capacity)
	def add_multiple_edges(self, node_is, node_js, capacities, rev_capacities):
		self.thisptr.add_multiple_edges(node_is, node_js, capacities, rev_capacities)

	def get_sink_nodes(self):
		return self.thisptr.get_sink_nodes()

	def maxflow(self):
		return self.thisptr.maxflow()
