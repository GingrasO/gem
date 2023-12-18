
function get_H_imp(n::Int,Jup::AbstractMatrix,Jdn::AbstractMatrix,U::AbstractArray;perm=1:n)
    os=OpSum()
    for i in 1:size(Jup,1)
        for j in i:size(Jup,2)
            if Jup[i,j]!=0.0
                @assert J[i,j]==conj(J[j,i])
                os+=Jup[i,j],"Cdagup",perm[i],"Cup",perm[j]
            end
            if Jdn[i,j]!=0.0
                os+=Jdn[i,j],"Cdagdn",perm[i],"Cdn",perm[j]
            end
        end
    end
    for i in 1:size(U,1), j in 1:size(U,2), k in 1:size(U,3), l in 1:size(U,4)
        if !iszero(U[i,j,k,l])
            pref=0.5
        
            cU=U[i,j,k,l]
            os+=(pref*cU,"Cdagup",perm[i],"Cdagdn",perm[j],"Cdn",perm[l],"Cup",perm[k])
            os+=(pref*cU,"Cdagdn",perm[i],"Cdagup",perm[j],"Cup",perm[l],"Cdn",perm[k])
            os+=(pref*cU,"Cdagup",perm[i],"Cdagup",perm[j],"Cup",perm[l],"Cup",perm[k])
            os+=(pref*cU,"Cdagdn",perm[i],"Cdagdn",perm[j],"Cdn",perm[l],"Cdn",perm[k])
        end
    end
    return os
end

function get_H_imp(n::Int,J::AbstractMatrix,U::AbstractArray;perm=1:n)
    os=OpSum()
    ###hopping

    for i in 1:size(J,1)
        for j in i:size(J,2)
            if J[i,j]!=0.0
                @assert J[i,j]==conj(J[j,i])
                @show i,j,J[i,j]
                os+=J[i,j],"Cdagup",perm[i],"Cup",perm[j]
                os+=J[i,j],"Cdagdn",perm[i],"Cdn",perm[j]
                #os+=conj(J[i,j]),"Cdagup",perm[j],"Cup",perm[i]
                #os+=conj(J[i,j]),"Cdagdn",perm[j],"Cdn",perm[i]
            end
        end
    end
    
    for i in 1:size(U,1)
        for j in 1:size(U,2)
            for k in 1:size(U,3)
                for l in 1:size(U,4)
                    if !iszero(U[i,j,k,l])
                        @assert U[i,j,k,l]==conj(U[l,k,j,i])
                        pref=0.5
                        #if i==k || j==l
                        #    continue
                        #end
                        #if i>k || i>l #|| j>k || j>l
                        #    pref=-1
                        #end
                        #if i==j==k==l
                            
                            cU=U[i,j,k,l]
                            os+=(pref*cU,"Cdagup",perm[i],"Cdagdn",perm[j],"Cdn",perm[l],"Cup",perm[k])
                            os+=(pref*cU,"Cdagdn",perm[i],"Cdagup",perm[j],"Cup",perm[l],"Cdn",perm[k])
                            os+=(pref*cU,"Cdagup",perm[i],"Cdagup",perm[j],"Cup",perm[l],"Cup",perm[k])
                            os+=(pref*cU,"Cdagdn",perm[i],"Cdagdn",perm[j],"Cdn",perm[l],"Cdn",perm[k])
                        #end
                        #os+=conj(U[i,j,k,l]),"Cdagdn",perm[l],"Cdagup",perm[k],"Cup",perm[j],"Cdn",perm[i]
                    end
                end
            end
        end
    end
    
    return os
end

function get_H_bath(n,Gamma;bathoffset=0,perm=1:n)
    os=OpSum()
    bo=bathoffset
    for i in 1:size(Gamma,1)
        for j in 1:size(Gamma,2)
            if !iszero(Gamma[i,j])
                os+=Gamma[i,j],"Cdagup",perm[bo+i],"Cup",perm[bo+j]
                os+=Gamma[i,j],"Cdagdn",perm[bo+i],"Cdn",perm[bo+j]
            end
        end
    end
    return os
end

function get_H_bath(n::Int,Gamma_up::AbstractMatrix,Gamma_dn::AbstractMatrix;bathoffset=0,perm=1:n)
    os=OpSum()
    bo=bathoffset
    for i in 1:size(Gamma_up,1)
        for j in 1:size(Gamma_up,2)
            if !iszero(Gamma[i,j])
                os+=Gamma_up[i,j],"Cdagup",perm[bo+i],"Cup",perm[bo+j]
                os+=Gamma_dn[i,j],"Cdagdn",perm[bo+i],"Cdn",perm[bo+j]
            end
        end
    end
    return os
end

function get_H_hyb(n,D;perm=1:n)
    os=OpSum()
    bo=size(D,1)
    for i in 1:size(D,1)
        for j in 1:size(D,2)
            if !iszero(D[i,j])
                os+=D[i,j],"Cdagup",perm[i],"Cup",perm[bo+j]
                os+=D[i,j],"Cdagdn",perm[i],"Cdn",perm[bo+j]
                os+=conj(D[i,j]),"Cdagup",perm[bo+j],"Cup",perm[i]
                os+=conj(D[i,j]),"Cdagdn",perm[bo+j],"Cdn",perm[i]
            end
        end
    end
    return os
end
         
function get_H_hyb(n,Dup::AbstractMatrix,Ddn::AbstractMatrix;perm=1:n)
    os=OpSum()
    bo=size(D,1)
    for i in 1:size(D,1)
        for j in 1:size(D,2)
            if !iszero(Dup[i,j])
                os+=Dup[i,j],"Cdagup",perm[i],"Cup",perm[bo+j]
                os+=conj(Dup[i,j]),"Cdagup",perm[bo+j],"Cup",perm[i]
            end
            if !iszero(Ddn[i,j])
                os+=Ddn[i,j],"Cdagdn",perm[i],"Cdn",perm[bo+j]
                os+=conj(Ddn[i,j]),"Cdagdn",perm[bo+j],"Cdn",perm[i]
            end
        end
    end
    return os
end

function get_S(n)
    os=OpSum()
    for i in 1:n
        os+=1.0, "Sz", i
        os+=1.0, "Sx",i
        os+=0.5*i, "Cdn",i,"Cdagup",i
        os+=-0.5*i, "Cup",i,"Cdagdn",i
    end
    return os
end


function get_Sz(n)
    os=OpSum()
    for i in 1:n
        os+=1.0, "Sz", i
    end
    return os
end

function get_Splus(n)
    os=OpSum()
    for i in 1:n
        os+=1.0, "S+", i
    end
    return os
end


function get_Sminus(n)
    os=OpSum()
    for i in 1:n
        os+=1.0, "S-", i
    end
    return os
end


function get_Ssquared(n)
    os=OpSum()
    for i in 1:n
        for j in 1:n
            os+=0.25, "Cdagup", i, "Cup",i , "Cdagup", j, "Cup",j
            os+=0.25, "Cdagdn", i, "Cdn",i , "Cdagdn", j, "Cdn",j
            os-=0.25, "Cdagdn", i, "Cdn",i , "Cdagup", j, "Cup",j
            os-=0.25, "Cdagup", i, "Cup",i , "Cdagdn", j, "Cdn",j
            
            
            os+=0.25, "Cup",i,"Cdagdn",i,"Cdn",j,"Cdagup",j
            os+=0.25,"Cup",j,"Cdagdn",j,"Cdn",i,"Cdagup",i
            os+=0.25, "Cdn",j,"Cdagup",j,"Cup",i,"Cdagdn",i
            os+=0.25, "Cdn",i,"Cdagup",i,"Cup",j,"Cdagdn",j
        end
    end
    return os
end