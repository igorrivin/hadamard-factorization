#!/bin/bash
# Run all tests for the Hadamard factorization project

echo "Hadamard Product Factorization Tests"
echo "===================================="
echo

echo "1. Running F2 search (this will take about 1 minute)..."
python hadamard_search.py
echo

echo "2. Verifying integer counterexamples..."
python verify_integers.py
echo

echo "3. Testing real number factorization..."
python test_real.py
echo

echo "Tests complete!"