function strain = pixelstrain(varargin)

    %PATCHSTRAIN
    %
    %
    %
    
    % This Source Code Form is subject to the terms of the Mozilla Public
    % License, v. 2.0. If a copy of the MPL was not distributed with this
    % file, You can obtain one at http://mozilla.org/MPL/2.0/.
    %
    % Copyright (c) 2016 DENSEanalysis Contributors
    
        %% PARSE APPLICATION DATA
    
        errid = sprintf('%s:invalidInput',mfilename);
    
        % parse input data
        defapi = struct(...
            'X',                [],...
            'Y',                [],...
            'mask',             [],...
            'times',            [],...
            'method',           'RSTLS',... %SG
            'dXt',              [],...  %SG
            'dYt',              [],...  %SG
            'Origin',           [],...
            'Orientation',      []);
        api = parseinputs(defapi,[],varargin{:});
    
        % check X/Y/mask
        X = api.X;
        Y = api.Y;
        mask = api.mask;
    
        % check times
        time = api.times;
    
        % additional parameters
        Ntime = numel(time);
        Isz   = [size(mask,1), size(mask,2)];
    
    
        % pixel trajectories
        % note this effectively checks the spline data as well
            xtrj = NaN([Isz,Ntime]);
            ytrj = xtrj;
                for k = 1:Ntime
                    dx = api.dXt(:,:,k);
                    dy = api.dYt(:,:,k);
                    xtrj(:,:,k) = X + reshape(dx,Isz);
                    ytrj(:,:,k) = Y + reshape(dy,Isz);
                end
    
    
        % parse origin
        origin = api.Origin;
        if isempty(origin)
            origin = [mean(X(mask)), mean(Y(mask))];
        end
    
        % parse orientation
        theta = api.Orientation;
        if isempty(theta)
            theta = cart2pol(X-origin(1),Y-origin(2));
        end
    
        % cos/sin calculation (saving computational effort)
        ct = cos(theta);
        st = sin(theta);
    
        % eliminate any invalid mask locations
        h = [0 1 0; 1 0 0; 0 0 0];
        tmp = false([Isz,4]);
        for k = 1:4
            tmp(:,:,k) = conv2(double(mask),h,'same')==2;
            h = rot90(h);
        end
    
        mask = any(tmp,3) & mask;
    
    
        %% STRAIN CALCULATION
    
        % determine number of neighbors
        h = [0 1 0; 1 0 1; 0 1 0];
        Nneighbor = mask.*conv2(double(mask),h,'same');
    
        % initialize output strain structure
        tmp = NaN([Isz Ntime]);
        strain = struct(...
            'vertices',     [],...
            'faces',        [],...
            'orientation',  [],...
            'maskimage',    [],...
            'XX',       tmp,...
            'YY',       tmp,...
            'XY',       tmp,...
            'YX',       tmp,...
            'RR',       tmp,...
            'CC',       tmp,...
            'RC',       tmp,...
            'CR',       tmp,...
            'p1',       tmp,...
            'p2',       tmp,...
            'p1or',     tmp);
    
        % strain calculation at each point
        dx = zeros(2,4);
        dX = zeros(2,4);
        tf = false(1,4);
    
        for fr = 1:Ntime
            if fr == 2
                a = 1;
            end
    
            for j = 1:Isz(2)
                for i = 1:Isz(1)
                    if mask(i,j) && Nneighbor(i,j) > 1
    
                        dx(:) = 0;
                        dX(:) = 0;
                        tf(:) = false;
    
                        if (i-1 >= 1) && mask(i-1,j)
                            tf(1) = true;
                            dx(:,1) = [xtrj(i-1,j,fr) - xtrj(i,j,fr);...
                                       ytrj(i-1,j,fr) - ytrj(i,j,fr)];
                            dX(:,1) = [X(i-1,j) - X(i,j); ...
                                       Y(i-1,j) - Y(i,j)];
                        end
    
                        if (i+1 <= Isz(1)) && mask(i+1,j)
                            tf(2) = true;
                            dx(:,2) = [xtrj(i+1,j,fr) - xtrj(i,j,fr);...
                                       ytrj(i+1,j,fr) - ytrj(i,j,fr)];
                            dX(:,2) = [X(i+1,j) - X(i,j); ...
                                       Y(i+1,j) - Y(i,j)];
                        end
    
                        if (j-1 >= 1) && mask(i,j-1)
                            tf(3) = true;
                            dx(:,3) = [xtrj(i,j-1,fr) - xtrj(i,j,fr);...
                                       ytrj(i,j-1,fr) - ytrj(i,j,fr)];
                            dX(:,3) = [X(i,j-1) - X(i,j); ...
                                       Y(i,j-1) - Y(i,j)];
                        end
    
                        if (j+1 <= Isz(2)) && mask(i,j+1)
                            tf(4) = true;
                            dx(:,4) = [xtrj(i,j+1,fr) - xtrj(i,j,fr);...
                                       ytrj(i,j+1,fr) - ytrj(i,j,fr)];
                            dX(:,4) = [X(i,j+1) - X(i,j); ...
                                       Y(i,j+1) - Y(i,j)];
                        end
    
    
                        % average deformation gradient tensor
                        Fave = dx(:,tf)/dX(:,tf);
    
                        % x/y strain tensor
                        E = 0.5*(Fave'*Fave - eye(2));
    
                        % coordinate system rotation matrix
                        % (Note this is the transpose of the vector rotation matrix)
                        Rot = [ct(i,j) st(i,j); -st(i,j) ct(i,j)];
    
                        % radial/circumferential strain tensor
                        Erot = Rot*E*Rot';
    
                        % principal strains
                        [v,d] = eig(E,'nobalance');
    
                        % record output
                        strain.XX(i,j,fr) = E(1);
                        strain.XY(i,j,fr) = E(2);
                        strain.YX(i,j,fr) = E(3);
                        strain.YY(i,j,fr) = E(4);
    
                        strain.RR(i,j,fr) = Erot(1);
                        strain.RC(i,j,fr) = Erot(2);
                        strain.CR(i,j,fr) = Erot(3);
                        strain.CC(i,j,fr) = Erot(4);
    
                        if all(d == 0)
                            strain.p1(i,j,fr) = 0;
                            strain.p2(i,j,fr) = 0;
                        else
                            strain.p1(i,j,fr)   = d(end);
                            strain.p2(i,j,fr)   = d(1);
                            strain.p1or(i,j,fr) = atan2(v(2,2),v(1,2));
                            % strain.p2or(i,j,fr) = atan2(v(2,1),v(1,1));
                        end
    
                    end
    
                end
    
            end
    
            if 1 == 1
                b = 2;
            end
        end
    end
    
    
    %% END OF FILE=============================================================
    