// A package for solving linear programming problems using the simplex method
package simplexsolve

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/mat"
)

// Errors we expect to use
var ERR_UNBOUNDED = errors.New("LP is unbounded")
var ERR_UNSOLVED = errors.New("LP is unsolved")

// Constraint represents a less-than-or-equal constraint in an LP.
// Coefficients stores the coefficients of the left hand side of the inequality,
// while RightHandSide is the value on the right hand side of the inequality.
// Example: 2x - 3y <= 15 ~ Constraint{[]float64{2, -3}, 15}
type Constraint struct {
	Coefficients  []float64
	RightHandSide float64
}

// Objective represents the objective function of an LP.
// Its entries are the coefficients of the function, and it is assumed
// that the function is being maximized.
type Objective []float64

// Tableau is a matrix that can be manipulated as the tableau
// used in simplex method calculations.
type Tableau struct {
	*mat.Dense
}

// NewTableau takes the Constraints and Objective that fully describe
// an LP and creates a Tableau describing that LP.
func NewTableau(cs []Constraint, o Objective) (*Tableau, error) {
	vecs := []float64{}
	for i, c := range cs {
		if len(c.Coefficients) != len(o) {
			return nil, errors.New("coefficient vectors must be of equal length")
		}

		slack := make([]float64, len(cs))
		slack[i] = 1

		vecs = append(vecs, append(c.Coefficients, append(slack, c.RightHandSide)...)...)
	}
	vecs = append(vecs, append(o, make([]float64, len(cs)+1)...)...)

	t := mat.NewDense(len(cs)+1, len(o)+len(cs)+1, vecs)
	return &Tableau{t}, nil
}

// Solve iterates operations of the simplex method until the LP represented by
// the Tableau t is solved or found to be unbounded.
func (t *Tableau) Solve() error {
	for !t.isSolved() {
		if t.isUnbounded() {
			return ERR_UNBOUNDED
		}
		t.scaleAndElim()
	}
	return nil
}

// ReadSoln reads the coefficients of the solution and value of the objective
// from a Tableau. Returns an error if the LP represented by the Tableau t
// is unsolved or unbounded.
func (t *Tableau) ReadSoln() ([]float64, float64, error) {
	if !t.isSolved() {
		return nil, 0, ERR_UNSOLVED
	}
	if t.isUnbounded() {
		return nil, 0, ERR_UNBOUNDED
	}
	nrs, ncs := t.Dims()
	nvs := ncs - nrs
	vals := make([]float64, nvs)
	for j := 0; j < nvs; j++ {
		for i := 0; i < nrs-1; i++ {
			if t.At(i, j) == 1 {
				vals[j] = t.At(i, ncs-1)
				break
			}
		}
	}

	return vals, t.At(nrs-1, ncs-1), nil
}

// Checks objective row for negative values. If there are none, LP is solved.
func (t *Tableau) isSolved() bool {
	nrs, _ := t.Dims()
	for _, v := range t.RawRowView(nrs - 1) {
		if v < 0 {
			return false
		}
	}
	return true
}

// Checks if an exiting variable can be found. If not, LP is unbounded.
func (t *Tableau) isUnbounded() bool {
	pr, _ := t.pivotIdxs()
	return pr < 0
}

// Finds the pivot element, then solves (1 iteration of simplex method)
func (t *Tableau) scaleAndElim() {
	pr, pc := t.pivotIdxs()
	t.scalePivotRow(pr, pc)
	t.elimRows(pr, pc)
}

func (t *Tableau) scalePivotRow(pr int, pc int) {
	// scale pivot row so pivot element is 1
	pe := t.At(pr, pc)
	row := t.RawRowView(pr)
	t.SetRow(pr, scaleSlice(row, 1/pe))
}

func (t *Tableau) elimRows(pr int, pc int) {
	nrs, _ := t.Dims()
	pivrow := t.RawRowView(pr)

	// for each row r other than the pivot row,
	// scale *the pivot row* and subtract it from r
	// so that r's entry in the pivot col becomes 0
	for i := 0; i < nrs; i++ {
		if i == pr {
			continue
		}
		t.SetRow(i, addSlices(t.RawRowView(i), scaleSlice(pivrow, -t.At(i, pc))))
	}
}

// Gets indices of the pivot row and pivot column
func (t *Tableau) pivotIdxs() (pr int, pc int) {
	nrs, ncs := t.Dims()

	// find pivot column index
	pc = negMinIdx(t.RowView(nrs - 1))

	// find pivot row index (min quotient with rightmost col)
	pr = -1
	minq := math.MaxFloat64
	for r := 0; r < nrs-1; r++ {
		pce := t.At(r, pc)
		if pce == 0 {
			continue
		}
		q := t.At(r, ncs-1) / pce
		if q < minq && q > 0 {
			pr = r
			minq = q
		}
	}
	return
}

// Gets the index of the min value in a vector that's negative
func negMinIdx(vec mat.Vector) int {
	mi := -1
	mv := float64(0)
	for i := 0; i < vec.Len()-1; i++ {
		if vec.AtVec(i) < mv {
			mi = i
			mv = vec.AtVec(i)
		}
	}
	return mi
}

// zip with for floats
func zipWithFloats(fn func(float64, float64) float64, sl1, sl2 []float64) []float64 {
	ret := make([]float64, len(sl1))
	for i := 0; i < len(sl1); i++ {
		ret[i] = fn(sl1[i], sl2[i])
	}
	return ret
}

// map for floats
func mapFloats(fn func(float64) float64, sl []float64) []float64 {
	ret := make([]float64, len(sl))
	for i, v := range sl {
		ret[i] = fn(v)
	}
	return ret
}

// Scales all elements in a slice of floats by a given float
func scaleSlice(sl []float64, s float64) []float64 {
	return mapFloats(func(v float64) float64 { return v * s }, sl)
}

// Adds corresponding elements of two slices
func addSlices(sl1, sl2 []float64) []float64 {
	return zipWithFloats(func(u, v float64) float64 { return u + v }, sl1, sl2)
}
