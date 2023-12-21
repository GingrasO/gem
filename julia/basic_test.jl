using PyCall

print_me(X::AbstractMatrix)=println(X)
double_me(X::AbstractMatrix)=2 .* X 
