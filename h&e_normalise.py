import os
import cv2 
import numpy as np
def normalise(true, false):
    Io = 240  # Normalising factor for intensity
    alpha = 1
    beta = 0.15
    for tile in os.listdir(true):
        tile = cv2.imread(f"{true}/{tile}")
        rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

        ######## Step 1: Convert RGB to OD ###################
        ## reference H&E OD matrix.
        HERef = np.array([[0.5626, 0.2159],
                        [0.7201, 0.8012],
                        [0.4062, 0.5581]])
        ### reference maximum stain concentrations for H&E
        maxCRef = np.array([1.9705, 1.0308])
        h, w, c = rgb.shape
        rgb = rgb.reshape(((-1, 3)))
        OD = -np.log10((rgb.astype(float)+1)/Io) 

        ############ Step 2: Remove data with OD intensity less than β ############
        # remove transparent pixels (clear region with no tissue)
        ODhat = OD[~np.any(OD < beta, axis=1)] #Returns an array where OD values are above beta

        ############# Step 3: Calculate SVD on the OD tuples ######################
        #Estimate covariance matrix of ODhat (transposed)
        # and then compute eigen values & eigenvectors.
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

        ######## Step 4: Create plane from the SVD directions with two largest values ######
        #project on the plane spanned by the eigenvectors corresponding to the two 
        # largest eigenvalues    
        That = ODhat.dot(eigvecs[:,1:3]) #Dot product

        ############### Step 5: Project data onto the plane, and normalize to unit length ###########
        ############## Step 6: Calculate angle of each point wrt the first SVD direction ########
        #find the min and max vectors and project back to OD space
        phi = np.arctan2(That[:,1],That[:,0])

        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100-alpha)

        vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)


        # a heuristic to make the vector corresponding to hematoxylin first and the 
        # one corresponding to eosin second
        if vMin[0] > vMax[0]:    
            HE = np.array((vMin[:,0], vMax[:,0])).T
            
        else:
            HE = np.array((vMax[:,0], vMin[:,0])).T


        # rows correspond to channels (RGB), columns to OD values
        Y = np.reshape(OD, (-1, 3)).T

        # determine concentrations of the individual stains
        C = np.linalg.lstsq(HE,Y, rcond=None)[0]

        # normalize stain concentrations
        maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
        tmp = np.divide(maxC,maxCRef)
        C2 = np.divide(C,tmp[:, np.newaxis])

        ###### Step 8: Convert extreme values back to OD space
        # recreate the normalized image using reference mixing matrix 

        Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
        Inorm[Inorm>255] = 254
        Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  

        # Separating H and E components

        H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
        H[H>255] = 254
        H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

        E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
        E[E>255] = 254
        E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
        
        cv2.imshow("normalised", Inorm)
        cv2.waitKey(0)
        cv2.imshow("seperated_H", H)
        cv2.waitKey(0)
        cv2.imshow("seperated_E", E)
        cv2.waitKey(0)

        
    for tile in os.listdir(true):
        tile = cv2.imread(f"{true}/{tile}")
        rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

        ######## Step 1: Convert RGB to OD ###################
        ## reference H&E OD matrix.
        HERef = np.array([[0.5626, 0.2159],
                        [0.7201, 0.8012],
                        [0.4062, 0.5581]])
        ### reference maximum stain concentrations for H&E
        maxCRef = np.array([1.9705, 1.0308])
        h, w, c = rgb.shape
        rgb = rgb.reshape(((-1, 3)))
        OD = -np.log10((rgb.astype(float)+1)/Io) 

        ############ Step 2: Remove data with OD intensity less than β ############
        # remove transparent pixels (clear region with no tissue)
        ODhat = OD[~np.any(OD < beta, axis=1)] #Returns an array where OD values are above beta

        ############# Step 3: Calculate SVD on the OD tuples ######################
        #Estimate covariance matrix of ODhat (transposed)
        # and then compute eigen values & eigenvectors.
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

        ######## Step 4: Create plane from the SVD directions with two largest values ######
        #project on the plane spanned by the eigenvectors corresponding to the two 
        # largest eigenvalues    
        That = ODhat.dot(eigvecs[:,1:3]) #Dot product

        ############### Step 5: Project data onto the plane, and normalize to unit length ###########
        ############## Step 6: Calculate angle of each point wrt the first SVD direction ########
        #find the min and max vectors and project back to OD space
        phi = np.arctan2(That[:,1],That[:,0])

        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100-alpha)

        vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)


        # a heuristic to make the vector corresponding to hematoxylin first and the 
        # one corresponding to eosin second
        if vMin[0] > vMax[0]:    
            HE = np.array((vMin[:,0], vMax[:,0])).T
            
        else:
            HE = np.array((vMax[:,0], vMin[:,0])).T


        # rows correspond to channels (RGB), columns to OD values
        Y = np.reshape(OD, (-1, 3)).T

        # determine concentrations of the individual stains
        C = np.linalg.lstsq(HE,Y, rcond=None)[0]

        # normalize stain concentrations
        maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
        tmp = np.divide(maxC,maxCRef)
        C2 = np.divide(C,tmp[:, np.newaxis])

        ###### Step 8: Convert extreme values back to OD space
        # recreate the normalized image using reference mixing matrix 

        Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
        Inorm[Inorm>255] = 254
        Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  

        # Separating H and E components

        H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
        H[H>255] = 254
        H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

        E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
        E[E>255] = 254
        E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
        
        cv2.imshow("normalised", Inorm)
        cv2.waitKey(0)
        cv2.imshow("seperated_H", H)
        cv2.waitKey(0)
        cv2.imshow("seperated_E", E)
        cv2.waitKey(0)


def main():
    directory_true = "TRUE_Group_for_Microsatellite_Instability_dMMR_MSI/Tiles"
    directory_false = "FALSE_Group_for_Microsatellite_Instability_pMMR_MSS/Tiles"
    normalise(directory_true, directory_false)



        

if __name__ == "__main__":
    main()