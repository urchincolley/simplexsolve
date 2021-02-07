package main

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/mat"
)

type Constraint struct {
	Coefficients  []float64
	RightHandSide float64
}

type Objective []float64

type Tableau struct {
	*mat.Dense
}

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

func (t *Tableau) Solve() {
	for !t.isSolved() {
		t.scaleAndElim()
	}
}

func (t *Tableau) ReadSoln() ([]float64, float64) {
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

	return vals, t.At(nrs-1, ncs-1)
}

func (t *Tableau) isSolved() bool {
	nrs, _ := t.Dims()
	for _, v := range t.RawRowView(nrs - 1) {
		if v < 0 {
			return false
		}
	}
	return true
}

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

func (t *Tableau) pivotIdxs() (pr int, pc int) {
	nrs, ncs := t.Dims()

	// find pivot column index
	pc = negMinIdx(t.RowView(nrs - 1))

	// find pivot row index (min quotient with rightmost col)
	minq := math.MaxFloat64
	for r := 0; r < nrs-1; r++ {
		q := t.At(r, ncs-1) / t.At(r, pc)
		if q < minq {
			pr = r
			minq = q
		}
	}
	return
}

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

func zipWithFloats(fn func(float64, float64) float64, sl1, sl2 []float64) []float64 {
	ret := make([]float64, len(sl1))
	for i := 0; i < len(sl1); i++ {
		ret[i] = fn(sl1[i], sl2[i])
	}
	return ret
}

func mapOverFloats(fn func(float64) float64, sl []float64) []float64 {
	ret := make([]float64, len(sl))
	for i, v := range sl {
		ret[i] = fn(v)
	}
	return ret
}

func scaleSlice(sl []float64, s float64) []float64 {
	return mapOverFloats(func(v float64) float64 { return v * s }, sl)
}

func addSlices(sl1, sl2 []float64) []float64 {
	return zipWithFloats(func(u, v float64) float64 { return u + v }, sl1, sl2)
}
