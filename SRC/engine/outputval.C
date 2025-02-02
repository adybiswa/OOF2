// -*- C++ -*-

/* This software was produced by NIST, an agency of the U.S. government,
 * and by statute is not subject to copyright in the United States.
 * Recipients of this software assume all responsibilities associated
 * with its operation, modification and maintenance. However, to
 * facilitate maintenance we ask that before distributing modified
 * versions of this software, you first contact the authors at
 * oof_manager@nist.gov. 
 */

#include <oofconfig.h>

#include "common/printvec.h"
#include "engine/fieldindex.h"
#include "engine/ooferror.h"
#include "engine/outputval.h"
#include <math.h>
#include <string.h>		// for memcpy

std::string ScalarOutputVal::classname_("ScalarOutputVal");
std::string VectorOutputVal::classname_("VectorOutputVal");

OutputValue::OutputValue(OutputVal *v)
  : val(v)
{
  ++val->refcount;
}

OutputValue::OutputValue(const OutputValue &other)
  : val(other.val)
{
  ++val->refcount;
}

OutputValue::~OutputValue() {
  if(--val->refcount == 0) {
    delete val;
  }
}

ArithmeticOutputValue::ArithmeticOutputValue(ArithmeticOutputVal *v)
  : OutputValue(v)
{}

NonArithmeticOutputValue::NonArithmeticOutputValue(NonArithmeticOutputVal *v)
  : OutputValue(v)
{}

const ArithmeticOutputValue &ArithmeticOutputValue::operator+=(
				       const ArithmeticOutputValue &other)
{
  ArithmeticOutputVal *thisval = dynamic_cast<ArithmeticOutputVal*>(val);
  const ArithmeticOutputVal *thatval =
    dynamic_cast<const ArithmeticOutputVal*>(other.val);
  *thisval += *thatval;
  return *this;
}

const ArithmeticOutputValue &ArithmeticOutputValue::operator-=(
				       const ArithmeticOutputValue &other)
{
  ArithmeticOutputVal *thisval = dynamic_cast<ArithmeticOutputVal*>(val);
  const ArithmeticOutputVal *thatval =
    dynamic_cast<const ArithmeticOutputVal*>(other.val);
  *thisval -= *thatval;
  return *this;
}

const ArithmeticOutputValue &ArithmeticOutputValue::operator *=(double x) {
  ArithmeticOutputVal *thisval = dynamic_cast<ArithmeticOutputVal*>(val);
  *thisval *= x;
  return *this;
}

double ArithmeticOutputValue::operator[](const FieldIndex &fi) const {
  const ArithmeticOutputVal *thisval =
    dynamic_cast<const ArithmeticOutputVal*>(val);
  return (*thisval)[fi];
}

double &ArithmeticOutputValue::operator[](const FieldIndex &fi) {
  ArithmeticOutputVal *thisval = dynamic_cast<ArithmeticOutputVal*>(val);
  return (*thisval)[fi];
}

ArithmeticOutputValue operator*(double x, const ArithmeticOutputValue &ov) {
  ArithmeticOutputValue result(ov);
  result *= x;
  return result;
}

ArithmeticOutputValue operator*(const ArithmeticOutputValue &ov, double x) {
  ArithmeticOutputValue result(ov);
  result *= x;
  return result;
}

ArithmeticOutputValue operator/(const ArithmeticOutputValue &ov, double x) {
  ArithmeticOutputValue result(ov);
  result *= 1./x;
  return result;
}

ArithmeticOutputValue operator+(const ArithmeticOutputValue &a,
				const ArithmeticOutputValue &b)
{
  ArithmeticOutputValue result(a);
  result += b;
  return result;
}

ArithmeticOutputValue operator-(const ArithmeticOutputValue &a,
				const ArithmeticOutputValue &b)
{
  ArithmeticOutputValue result(a);
  result -= b;
  return result;
}

//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//

// These are here just so that debugging lines can be added if needed.

OutputVal::OutputVal() : refcount(0) {}

OutputVal::~OutputVal() {}

//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//

const ScalarOutputVal &ScalarOutputVal::operator=(const OutputVal &other) {
  *this = dynamic_cast<const ScalarOutputVal&>(other);
  return *this;
}

const ScalarOutputVal &ScalarOutputVal::operator=(const ScalarOutputVal &other)
{
  val = other.val;
  return *this;
}

double ScalarOutputVal::operator[](const FieldIndex&) const {
  return val;
}

double &ScalarOutputVal::operator[](const FieldIndex&) {
  return val;
}

std::vector<double> *ScalarOutputVal::value_list() const {
  return new std::vector<double>(1, val);
}


FieldIndex *ScalarOutputVal::getIndex(const std::string&) const {
  return new ScalarFieldIndex();
}

ComponentsP ScalarOutputVal::components() const {
  static const ScalarFieldComponents comps;
  return ComponentsP(&comps);
}

ScalarOutputVal operator+(const ScalarOutputVal &a, const ScalarOutputVal &b) {
  ScalarOutputVal result(a);
  result += b;
  return result;
}

ScalarOutputVal operator-(const ScalarOutputVal &a, const ScalarOutputVal &b) {
  ScalarOutputVal result(a);
  result -= b;
  return result;
}

ScalarOutputVal operator*(const ScalarOutputVal &a, double b) {
  ScalarOutputVal result(a);
  result *= b;
  return result;
}

ScalarOutputVal operator*(double b, const ScalarOutputVal &a) {
  ScalarOutputVal result(a);
  result *= b;
  return result;
}

ScalarOutputVal operator/(const ScalarOutputVal &a, double b) {
  ScalarOutputVal result(a);
  result *= (1./b);
  return result;
}

//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//

VectorOutputVal::VectorOutputVal()
  : size_(0),
    components_(nullptr),
    data(nullptr)
{}

VectorOutputVal::VectorOutputVal(int n)
  : size_(n),
    components_(nullptr),
    data(new double[n])
{
  for(int i=0; i<n; i++)
    data[i] = 0.0;
}

VectorOutputVal::VectorOutputVal(const VectorOutputVal &other)
  : size_(other.size_),
    components_(nullptr),
    data(new double[other.size_])
{
  (void) memcpy(data, other.data, size_*sizeof(double));
}

VectorOutputVal::VectorOutputVal(const std::vector<double> &vec)
  : size_(vec.size()),
    components_(nullptr),
    data(new double[vec.size()])
{
  (void) memcpy(data, vec.data(), size_*sizeof(double));
}

VectorOutputVal::~VectorOutputVal() {
  delete [] data;
  delete components_;
}

const VectorOutputVal &VectorOutputVal::operator=(const OutputVal &other) {
  *this = dynamic_cast<const VectorOutputVal&>(other);
  delete components_;
  return *this;
}

const VectorOutputVal &VectorOutputVal::operator=(const VectorOutputVal &other)
{
  delete [] data;
  delete components_;
  components_ = nullptr;
  size_ = other.size();
  data = new double[size_];
  (void) memcpy(data, other.data, size_*sizeof(double));
  return *this;
}

std::vector<double> *VectorOutputVal::value_list() const {
  std::vector<double> *res = new std::vector<double>(size_);
  (void) memcpy(res->data(), data, size_*sizeof(double));
  return res;
}

VectorOutputVal *VectorOutputVal::clone() const {
  return new VectorOutputVal(*this);
}

VectorOutputVal *VectorOutputVal::zero() const {
  return new VectorOutputVal(size());
}

VectorOutputVal *VectorOutputVal::one() const {
  VectorOutputVal *won = new VectorOutputVal(size());
  for(unsigned int i=0; i<size(); i++)
    won->data[i] = 1.0;
  return won;
}

double VectorOutputVal::dot(const std::vector<double> &other) const {
  assert(size() == other.size());
  double sum = 0;
  for(unsigned int i=0; i<size(); i++)
    sum += data[i]*other[i];
  return sum;
}

ComponentsP VectorOutputVal::components() const {
  // Because ComponentsP doesn't delete its Components, we can't
  // allocate and return a new Components object each time this is
  // called.  But we can't create a Components object in the
  // VectorOutputVal constructor, because we might not know the size
  // yet.
  if(components_ == nullptr ||
     components_->min() != 0 || components_->max() != size_)
    {
      delete components_;
      components_ = new VectorFieldComponents(0, size_);
    }
  return ComponentsP(components_);
}

double VectorOutputVal::operator[](const FieldIndex &fi) const {
  return data[fi.integer()];
}

double &VectorOutputVal::operator[](const FieldIndex &fi) {
  return data[fi.integer()];
}

FieldIndex *VectorOutputVal::getIndex(const std::string &str) const {
  // TODO: Specify the names of the components in the constructor?
  // Not all 2D or 3D vectors are space vectors.
  if(size_ <= 3) {
    // str must be "x", "y", or "z"
    return new VectorFieldIndex(str[0] - 'x');
  }
  // str must be "0", "1", "2", etc.
  return new VectorFieldIndex(str[0] - '0');
}

double VectorOutputVal::magnitude() const {
  double sum = 0.0;
  for(unsigned int i=0; i<size_; i++) {
    double x = data[i];
    sum += x*x;
  }
  return sqrt(sum);
}

VectorOutputVal operator+(const VectorOutputVal &a, const VectorOutputVal &b) {
  VectorOutputVal result(a);
  result += b;
  return result;
}

VectorOutputVal operator-(const VectorOutputVal &a, const VectorOutputVal &b) {
  VectorOutputVal result(a);
  result -= b;
  return result;
}

VectorOutputVal operator*(const VectorOutputVal &a, double b) {
  VectorOutputVal result(a);
  result *= b;
  return result;
}

VectorOutputVal operator*(double b, const VectorOutputVal &a) {
  VectorOutputVal result(a);
  result *= b;
  return result;
}

VectorOutputVal operator/(const VectorOutputVal &a, double b) {
  VectorOutputVal result(a);
  result *= (1./b);
  return result;
}

//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//

std::string ListOutputVal::classname_("ListOutputVal");

ListOutputVal::ListOutputVal(const std::vector<std::string> *lbls)
  : size_(lbls->size()),
    data(new double[lbls->size()]),
    components_(this),
    labels(*lbls) 
{
  for(unsigned int i=0; i<size_; i++)
    data[i] = 0.0;
}

ListOutputVal::ListOutputVal(const std::vector<std::string> *lbls,
			     const std::vector<double> &vec)
  : size_(vec.size()),
    data(new double[size_]),
    components_(this),
    labels(*lbls)
{
  (void) memcpy(data, vec.data(), size_*sizeof(double));
}

ListOutputVal::ListOutputVal(const ListOutputVal &other)
  : size_(other.size()),
    data(new double[other.size()]),
    components_(this),
    labels(other.labels)
{
  (void) memcpy(data, other.data, size_*sizeof(double));
}

ListOutputVal::~ListOutputVal() {
  delete [] data;
}

ComponentsP ListOutputVal::components() const {
  return ComponentsP(&components_);
}

double ListOutputVal::operator[](const FieldIndex &fi) const {
  return data[fi.integer()];
}

double &ListOutputVal::operator[](const FieldIndex &fi) {
  return data[fi.integer()];
}

const ListOutputVal &ListOutputVal::operator=(const ListOutputVal &other) {
  delete [] data;
  size_ = other.size();
  data = new double[size_];
  (void) memcpy(data, other.data, size_*sizeof(double));
  return *this;
}

const ListOutputVal &ListOutputVal::operator=(const OutputVal &other) {
  *this = dynamic_cast<const ListOutputVal&>(other);
  return *this;
}

ListOutputVal *ListOutputVal::zero() const {
  return new ListOutputVal(&labels);
}

ListOutputVal *ListOutputVal::clone() const {
  return new ListOutputVal(*this);
}

std::vector<double> *ListOutputVal::value_list() const {
  std::vector<double> *res = new std::vector<double>(size_);
  memcpy(res->data(), data, size_*sizeof(data));
  return res;
}

FieldIndex *ListOutputVal::getIndex(const std::string &s) const {
  for(int i=0; i<size(); i++) {
    if(labels[i] == s)
      return new ListOutputValIndex(this, i);
  }
  throw ErrProgrammingError("Bad index '" + s + "'", __FILE__, __LINE__);
}

const std::string& ListOutputValIndex::classname() const {
  static const std::string nm("ListOutputValIndex");
  return nm;
}

const std::string &ListOutputValIndex::shortrepr() const {
  return ov_->labels[index_];
}

void ListOutputValIndex::print(std::ostream &os) const {
  os << "ListOutputValIndex(" << index_ << ")";
}

bool ListOutputValIterator::operator!=(const ComponentIterator &othr) const {
  const ListOutputValIterator& other =
    dynamic_cast<const ListOutputValIterator&>(othr);
  return other.ov_ != ov_ || other.v != v;
}

void ListOutputValIterator::print(std::ostream &os) const {
  os << "ListOutputValIterator(" << *ov_ << ")";
}

ComponentIteratorP ListOutputValComponents::begin() const {
  return ComponentIteratorP(new ListOutputValIterator(ov_, 0));
}

ComponentIteratorP ListOutputValComponents::end() const {
  return ComponentIteratorP(new ListOutputValIterator(ov_, ov_->size()));
}

//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//

std::ostream &operator<<(std::ostream &os, const OutputVal &ov) {
  ov.print(os);
  return os;
}

std::ostream &operator<<(std::ostream &os, const OutputValue &value) {
  return os << *value.val;
}

void ScalarOutputVal::print(std::ostream &os) const {
  os << "ScalarOutputVal(" << val << ")";
}

void VectorOutputVal::print(std::ostream &os) const {
  os << "VectorOutputVal(";
  if(size_ > 0) {
     os << data[0];
     for(unsigned int i=1; i<size_; i++) {
       os << ", " << data[i];
     }
  }
  os << ")";
}

void ListOutputVal::print(std::ostream &os) const {
  os << "ListOutputVal(";
  if(size_ > 0) {
    os << data[0];
    for(unsigned int i=1; i<size_; i++)
      os << ", " << data[i];
  }
  os << ")";
}

//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//

// For debugging

// const OutputVal &OutputVal::operator=(const OutputVal &other) {
//   std::cerr << "***** OutputVal::operator=: this=" << *this << std::endl;
//   std::cerr << "                           other=" << other << std::endl;
//   abort();
//   return *this;
// }
