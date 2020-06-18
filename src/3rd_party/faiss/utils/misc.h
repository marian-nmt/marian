#pragma once
#include <algorithm>
#include <vector>

namespace faiss {
  /// The metric space for vector comparison for Faiss indices and algorithms.
///
/// Most algorithms support both inner product and L2, with the flat
/// (brute-force) indices supporting additional metric types for vector
/// comparison.
  enum MetricType {
    METRIC_INNER_PRODUCT = 0,  ///< maximum inner product search
    METRIC_L2 = 1,             ///< squared L2 search
    METRIC_L1,                 ///< L1 (aka cityblock)
    METRIC_Linf,               ///< infinity distance
    METRIC_Lp,                 ///< L_p distance, p is given by a faiss::Index
                               /// metric_arg

    /// some additional metrics defined in scipy.spatial.distance
    METRIC_Canberra = 20,
    METRIC_BrayCurtis,
    METRIC_JensenShannon,
  };

  template<class T>
  struct ScopeDeleter {
    const T * ptr;
    explicit ScopeDeleter(const T* ptr = nullptr) : ptr(ptr) {}
    void release() { ptr = nullptr; }
    void set(const T * ptr_in) { ptr = ptr_in; }
    void swap(ScopeDeleter<T> &other) { std::swap(ptr, other.ptr); }
    ~ScopeDeleter() {
      delete[] ptr;
    }
  };

  //////////////////////////////////////////
  using idx_t = int64_t;

  /** List of temporary buffers used to store results before they are
 *  copied to the RangeSearchResult object. */
  struct BufferList {
    typedef faiss::idx_t idx_t;

    // buffer sizes in # entries
    size_t buffer_size;

    struct Buffer {
      idx_t *ids;
      float *dis;
    };

    std::vector<Buffer> buffers;
    size_t wp; ///< write pointer in the last buffer.

    explicit BufferList(size_t buffer_size);

    ~BufferList();

    /// create a new buffer
    void append_buffer();

    /// add one result, possibly appending a new buffer if needed
    void add(idx_t id, float dis);

    /// copy elemnts ofs:ofs+n-1 seen as linear data in the buffers to
    /// tables dest_ids, dest_dis
    void copy_range(size_t ofs, size_t n,
      idx_t * dest_ids, float *dest_dis);

  };

  /** The objective is to have a simple result structure while
 *  minimizing the number of mem copies in the result. The method
 *  do_allocation can be overloaded to allocate the result tables in
 *  the matrix type of a scripting language like Lua or Python. */
  struct RangeSearchResult {
    size_t nq;      ///< nb of queries
    size_t *lims;   ///< size (nq + 1)

    typedef faiss::idx_t idx_t;

    idx_t *labels;     ///< result for query i is labels[lims[i]:lims[i+1]]
    float *distances;  ///< corresponding distances (not sorted)

    size_t buffer_size; ///< size of the result buffers used

    /// lims must be allocated on input to range_search.
    explicit RangeSearchResult(idx_t nq, bool alloc_lims = true);

    /// called when lims contains the nb of elements result entries
    /// for each query

    virtual void do_allocation();

    virtual ~RangeSearchResult();
  };

  struct RangeSearchPartialResult;

  /// result structure for a single query
  struct RangeQueryResult {
    using idx_t = faiss::idx_t;
    idx_t qno;    //< id of the query
    size_t nres;  //< nb of results for this query
    RangeSearchPartialResult * pres;

    /// called by search function to report a new result
    void add(float dis, idx_t id);
  };

  /// the entries in the buffers are split per query
  struct RangeSearchPartialResult : BufferList {
    RangeSearchResult * res;

    /// eventually the result will be stored in res_in
    explicit RangeSearchPartialResult(RangeSearchResult * res_in);

    /// query ids + nb of results per query.
    std::vector<RangeQueryResult> queries;

    /// begin a new result
    RangeQueryResult & new_result(idx_t qno);

    /*****************************************
     * functions used at the end of the search to merge the result
     * lists */
    void finalize();

    /// called by range_search before do_allocation
    void set_lims();

    /// called by range_search after do_allocation
    void copy_result(bool incremental = false);

    /// merge a set of PartialResult's into one RangeSearchResult
    /// on ouptut the partialresults are empty!
    static void merge(std::vector <RangeSearchPartialResult *> &
      partial_results, bool do_delete = true);

  };


} // namespace
