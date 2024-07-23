function zgrid = DENSE_displacement_RSTLS(x,y,z,xnodes,ynodes,zf1,varargin)

    % set defaults
    params.smoothness = 4;
    params.Tempsmoothness = 0.5;
    params.interp = 'bilinear';
    params.solver = 'normal';
    params.maxiter = [];
    params.extend = 'warning';
    params.tilesize = inf;
    params.overlap = 0.20;
    params.mask = [];
    params.faster = false;
    
    
    % was the params struct supplied?
    if ~isempty(varargin)
        if isstruct(varargin{1})
            % params is only supplied if its a call from tiled_gridfit
            params = varargin{1};
            if length(varargin)>1
                % check for any overrides
                params = parse_pv_pairs(params,varargin{2:end});
            end
        else
            % check for any overrides of the defaults
            params = parse_pv_pairs(params,varargin);
            
        end
    end
    
    % check the parameters for acceptability
    params = check_params(params);
    
    % ensure all of x,y,z,xnodes,ynodes are column vectors,
    % also drop any NaN data
    x=x(:);
    y=y(:);
    z=z(:);
    k = isnan(x) | isnan(y) | isnan(z);
    if any(k)
        x(k)=[];
        y(k)=[];
        z(k)=[];
    end
    xmin = min(x);
    xmax = max(x);
    ymin = min(y);
    ymax = max(y);
    
    % did they supply a scalar for the nodes?
    if length(xnodes)==1
        xnodes = linspace(xmin,xmax,xnodes)';
        xnodes(end) = xmax; % make sure it hits the max
    end
    if length(ynodes)==1
        ynodes = linspace(ymin,ymax,ynodes)';
        ynodes(end) = ymax; % make sure it hits the max
    end
    
    xnodes=xnodes(:);
    ynodes=ynodes(:);
    dx = diff(xnodes);
    dy = diff(ynodes);
    nx = length(xnodes);
    ny = length(ynodes);
    ngrid = nx*ny;
    
    
    % its a single tile.
    
    % mask must be either an empty array, or a boolean
    % aray of the same size as the final grid.
    nmask = size(params.mask);
    if ~isempty(params.mask) && ((nmask(2)~=nx) || (nmask(1)~=ny))
        if ((nmask(2)==ny) || (nmask(1)==nx))
            error 'Mask array is probably transposed from proper orientation.'
        else
            error 'Mask array must be the same size as the final grid.'
        end
    end
    if ~isempty(params.mask)
        params.maskflag = 1;
    else
        params.maskflag = 0;
    end
    
    % default for maxiter?
    if isempty(params.maxiter)
        params.maxiter = min(10000,nx*ny);
    end
    
    % check lengths of the data
    n = length(x);
    if (length(y)~=n) || (length(z)~=n)
        error 'Data vectors are incompatible in size.'
    end
    if n<3
        error 'Insufficient data for surface estimation.'
    end
    
    % verify the nodes are distinct
    if any(diff(xnodes)<=0) || any(diff(ynodes)<=0)
        error 'xnodes and ynodes must be monotone increasing'
    end
    
    % Are there enough output points to form a surface?
    % Bicubic interpolation requires a 4x4 output grid.  Other types require a 3x3 output grid.
    if strcmp(params.interp, 'bicubic')
        MinAxisLength = 4;
    else
        MinAxisLength = 3;
    end
    if length(xnodes) < MinAxisLength
        error(['The output grid''s x axis must have at least ', num2str(MinAxisLength), ' nodes.']);
    end
    if length(ynodes) < MinAxisLength
        error(['The output grid''s y axis must have at least ', num2str(MinAxisLength), ' nodes.']);
    end
    clear MinAxisLength;
    
    % do we need to tweak the first or last node in x or y?
    if xmin<xnodes(1)
        switch params.extend
            case 'always'
                xnodes(1) = xmin;
            case 'warning'
                warning('GRIDFIT:extend',['xnodes(1) was decreased by: ',num2str(xnodes(1)-xmin),', new node = ',num2str(xmin)])
                xnodes(1) = xmin;
            case 'never'
                error(['Some x (',num2str(xmin),') falls below xnodes(1) by: ',num2str(xnodes(1)-xmin)])
        end
    end
    if xmax>xnodes(end)
        switch params.extend
            case 'always'
                xnodes(end) = xmax;
            case 'warning'
                warning('GRIDFIT:extend',['xnodes(end) was increased by: ',num2str(xmax-xnodes(end)),', new node = ',num2str(xmax)])
                xnodes(end) = xmax;
            case 'never'
                error(['Some x (',num2str(xmax),') falls above xnodes(end) by: ',num2str(xmax-xnodes(end))])
        end
    end
    if ymin<ynodes(1)
        switch params.extend
            case 'always'
                ynodes(1) = ymin;
            case 'warning'
                warning('GRIDFIT:extend',['ynodes(1) was decreased by: ',num2str(ynodes(1)-ymin),', new node = ',num2str(ymin)])
                ynodes(1) = ymin;
            case 'never'
                error(['Some y (',num2str(ymin),') falls below ynodes(1) by: ',num2str(ynodes(1)-ymin)])
        end
    end
    if ymax>ynodes(end)
        switch params.extend
            case 'always'
                ynodes(end) = ymax;
            case 'warning'
                warning('GRIDFIT:extend',['ynodes(end) was increased by: ',num2str(ymax-ynodes(end)),', new node = ',num2str(ymax)])
                ynodes(end) = ymax;
            case 'never'
                error(['Some y (',num2str(ymax),') falls above ynodes(end) by: ',num2str(ymax-ynodes(end))])
        end
    end
    
    % determine which cell in the array each point lies in
    [~, indx] = histc(x,xnodes);
    [~, indy] = histc(y,ynodes);
    % any point falling at the last node is taken to be
    % inside the last cell in x or y.
    k=(indx==nx);
    indx(k)=indx(k)-1;
    k=(indy==ny);
    indy(k)=indy(k)-1;
    ind = indy + ny*(indx-1);
    
    % Do we have a mask to apply?
    if params.maskflag
        % if we do, then we need to ensure that every
        % cell with at least one data point also has at
        % least all of its corners unmasked.
        params.mask(ind) = 1;
        params.mask(ind+1) = 1;
        params.mask(ind+ny) = 1;
        params.mask(ind+ny+1) = 1;
    end
    
    % interpolation equations for each point
    tx = double(min(1,max(0,(x - xnodes(indx))./dx(indx))));
    ty = double(min(1,max(0,(y - ynodes(indy))./dy(indy))));
    % Future enhancement: add cubic interpolant
    switch params.interp
        case 'triangle'
            % linear interpolation inside each triangle
            k = (tx > ty);
            L = ones(n,1);
            L(k) = ny;
            
            t1 = min(tx,ty);
            t2 = max(tx,ty);
            A = sparse(repmat((1:n)', 1, 3), [ind, ind + ny + 1, ind + L], [1 - t2, t1, t2 - t1], n, ngrid);
            
        case 'nearest'
            % nearest neighbor interpolation in a cell
            k = round(1-ty) + round(1-tx)*ny;
            A = sparse((1:n)',ind+k,ones(n,1),n,ngrid);
            
        case 'bilinear'
            % bilinear interpolation in a cell
            A = sparse(repmat((1:n)',1,4),[ind,ind+1,ind+ny,ind+ny+1], ...
                [(1-tx).*(1-ty), (1-tx).*ty, tx.*(1-ty), tx.*ty], ...
                n,ngrid);
    end
    %   Aint = A;
    rhs = z;
    
    
    % Build a regularizer using the second derivative.  This used to be called "gradient" even though it uses a second
    % derivative, not a first derivative.  This is an important distinction because "gradient" implies a horizontal
    % surface, which is not correct.  The second derivative favors flatness, especially if you use a large smoothness
    % constant.  Flat and horizontal are two different things, and in this script we are taking an irregular surface and
    % flattening it according to the smoothness constant.
    % The second-derivative calculation is documented here:
    % http://mathformeremortals.wordpress.com/2013/01/12/a-numerical-second-derivative-from-three-points/
    
    % Minimizes the sum of the squares of the second derivatives (wrt x and y) across the grid
    [i,j] = meshgrid(1:nx,2:(ny-1));
    ind = j(:) + ny*(i(:)-1);
    dy1 = dy(j(:)-1);
    dy2 = dy(j(:));
    
    Areg = sparse(repmat(ind,1,3),[ind-1,ind,ind+1], ...
        [-2./(dy1.*(dy1+dy2)), ...
        2./(dy1.*dy2), -2./(dy2.*(dy1+dy2))],ngrid,ngrid);
    
    [i,j] = meshgrid(2:(nx-1),1:ny);
    ind = j(:) + ny*(i(:)-1);
    dx1 = dx(i(:) - 1);
    dx2 = dx(i(:));
    
    Areg = [Areg;sparse(repmat(ind,1,3),[ind-ny,ind,ind+ny], ...
        [-2./(dx1.*(dx1+dx2)), ...
        2./(dx1.*dx2), -2./(dx2.*(dx1+dx2))],ngrid,ngrid)];
    nreg = size(Areg, 1);
    
    FidelityEquationCount = size(A, 1);
    % Number of the second derivative equations in the matrix
    RegularizerEquationCount = nx * (ny - 2) + ny * (nx - 2);
    % We are minimizing the sum of squared errors, so adjust the magnitude of the squared errors to make second-derivative
    % squared errors match the fidelity squared errors.  Then multiply by smoothparam.
    NewSmoothnessScale = params.smoothness * sqrt(FidelityEquationCount / RegularizerEquationCount);
    
    % Second derivatives scale with z exactly because d^2(K*z) / dx^2 = K * d^2(z) / dx^2.
    % That means we've taken care of the z axis.
    % The square root of the point/derivative ratio takes care of the grid density.
    
    
    if params.faster    
        scaledAreg = Areg * NewSmoothnessScale;
        A = [A; scaledAreg; speye(length(zf1(:))) * params.Tempsmoothness];
    else
        Atemp = sparse(eye(length(zf1(:))));
        % NewSmoothnessScale
        A = [A; Areg * NewSmoothnessScale; Atemp * params.Tempsmoothness];
    end
    
    rhs = double([rhs;zeros(nreg,1);params.Tempsmoothness*zf1(:)]);
    
    % do we have a mask to apply?
    if params.maskflag
        unmasked = find(params.mask);
    end
    % solve the full system, with regularizer attached
    switch params.solver
        case {'\' 'backslash'}
            if params.maskflag
                % there is a mask to use
                zgrid=nan(ny,nx);
                zgrid(unmasked) = A(:,unmasked)\rhs;
            else
                % no mask
                zgrid = reshape(A\rhs,ny,nx);
            end
            
        case 'normal'
            % The normal equations, solved with \. Can be faster
            % for huge numbers of data points, but reasonably
            % sized grids. The regularizer makes A well conditioned
            % so the normal equations are not a terribly bad thing
            % here.
            if params.maskflag
                % there is a mask to use
                Aunmasked = A(:,unmasked);
                zgrid=nan(ny,nx);
                zgrid(unmasked) = (Aunmasked'*Aunmasked)\(Aunmasked'*rhs);
            else
                zgrid = reshape((A'*A)\(A'*rhs),ny,nx);
            end
            
        case 'symmlq'
            % iterative solver - symmlq - requires a symmetric matrix,
            % so use it to solve the normal equations. No preconditioner.
            tol = abs(max(z)-min(z))*1.e-13;
            if params.maskflag
                % there is a mask to use
                zgrid=nan(ny,nx);
                [zgrid(unmasked),flag] = symmlq(A(:,unmasked)'*A(:,unmasked), ...
                    A(:,unmasked)'*rhs,tol,params.maxiter);
            else
                [zgrid,flag] = symmlq(A'*A,A'*rhs,tol,params.maxiter);
                zgrid = reshape(zgrid,ny,nx);
            end
            % display a warning if convergence problems
            switch flag
                case 0
                    % no problems with convergence
                case 1
                    % SYMMLQ iterated MAXIT times but did not converge.
                    warning('GRIDFIT:solver',['Symmlq performed ',num2str(params.maxiter), ...
                        ' iterations but did not converge.'])
                case 3
                    % SYMMLQ stagnated, successive iterates were the same
                    warning('GRIDFIT:solver','Symmlq stagnated without apparent convergence.')
                otherwise
                    warning('GRIDFIT:solver',['One of the scalar quantities calculated in',...
                        ' symmlq was too small or too large to continue computing.'])
            end
            
        case 'lsqr'
            % iterative solver - lsqr. No preconditioner here.
            tol = abs(max(z)-min(z))*1.e-13;
            if params.maskflag
                % there is a mask to use
                zgrid=nan(ny,nx);
                [zgrid(unmasked),flag] = lsqr(A(:,unmasked),rhs,tol,params.maxiter);
            else
                [zgrid,flag] = lsqr(A,rhs,tol,params.maxiter);
                zgrid = reshape(zgrid,ny,nx);
            end
            
            % display a warning if convergence problems
            switch flag
                case 0
                    % no problems with convergence
                case 1
                    % lsqr iterated MAXIT times but did not converge.
                    warning('GRIDFIT:solver',['Lsqr performed ', ...
                        num2str(params.maxiter),' iterations but did not converge.'])
                case 3
                    % lsqr stagnated, successive iterates were the same
                    warning('GRIDFIT:solver','Lsqr stagnated without apparent convergence.')
                case 4
                    warning('GRIDFIT:solver',['One of the scalar quantities calculated in',...
                        ' LSQR was too small or too large to continue computing.'])
            end
            
    end  % switch params.solver
    
    
    % only generate xgrid and ygrid if requested.
    if nargout>1
        [xgrid,ygrid]=meshgrid(xnodes,ynodes);
    end
    
    % ============================================
    % End of main function - gridfit
    % ============================================
    
    % ============================================
    % subfunction - parse_pv_pairs
    % ============================================
    function params=parse_pv_pairs(params,pv_pairs)
    % parse_pv_pairs: parses sets of property value pairs, allows defaults
    % usage: params=parse_pv_pairs(default_params,pv_pairs)
    %
    % arguments: (input)
    %  default_params - structure, with one field for every potential
    %             property/value pair. Each field will contain the default
    %             value for that property. If no default is supplied for a
    %             given property, then that field must be empty.
    %
    %  pv_array - cell array of property/value pairs.
    %             Case is ignored when comparing properties to the list
    %             of field names. Also, any unambiguous shortening of a
    %             field/property name is allowed.
    %
    % arguments: (output)
    %  params   - parameter struct that reflects any updated property/value
    %             pairs in the pv_array.
    %
    % Example usage:
    % First, set default values for the parameters. Assume we
    % have four parameters that we wish to use optionally in
    % the function examplefun.
    %
    %  - 'viscosity', which will have a default value of 1
    %  - 'volume', which will default to 1
    %  - 'pie' - which will have default value 3.141592653589793
    %  - 'description' - a text field, left empty by default
    %
    % The first argument to examplefun is one which will always be
    % supplied.
    %
    %   function examplefun(dummyarg1,varargin)
    %   params.Viscosity = 1;
    %   params.Volume = 1;
    %   params.Pie = 3.141592653589793
    %
    %   params.Description = '';
    %   params=parse_pv_pairs(params,varargin);
    %   params
    %
    % Use examplefun, overriding the defaults for 'pie', 'viscosity'
    % and 'description'. The 'volume' parameter is left at its default.
    %
    %   examplefun(rand(10),'vis',10,'pie',3,'Description','Hello world')
    %
    % params =
    %     Viscosity: 10
    %        Volume: 1
    %           Pie: 3
    %   Description: 'Hello world'
    %
    % Note that capitalization was ignored, and the property 'viscosity'
    % was truncated as supplied. Also note that the order the pairs were
    % supplied was arbitrary.
    
    npv = length(pv_pairs);
    n = npv/2;
    
    if n~=floor(n)
        error 'Property/value pairs must come in PAIRS.'
    end
    if n<=0
        % just return the defaults
        return
    end
    
    if ~isstruct(params)
        error 'No structure for defaults was supplied'
    end
    
    % there was at least one pv pair. process any supplied
    propnames = fieldnames(params);
    lpropnames = lower(propnames);
    for i=1:n
        p_i = lower(pv_pairs{2*i-1});
        v_i = pv_pairs{2*i};
        
        ind = strmatch(p_i,lpropnames,'exact');
        if isempty(ind)
            ind = find(strncmp(p_i,lpropnames,length(p_i)));
            if isempty(ind)
                error(['No matching property found for: ',pv_pairs{2*i-1}])
            elseif length(ind)>1
                error(['Ambiguous property name: ',pv_pairs{2*i-1}])
            end
        end
        p_i = propnames{ind};
        
        % override the corresponding default in params
        params = setfield(params,p_i,v_i); %#ok
        
    end
    
    
    % ============================================
    % subfunction - check_params
    % ============================================
    function params = check_params(params)
    
    % check the parameters for acceptability
    % smoothness == 1 by default
    if isempty(params.smoothness)
        params.smoothness = 1;
    else
        if (numel(params.smoothness)>2) || any(params.smoothness<=0)
            error 'Smoothness must be scalar (or length 2 vector), real, finite, and positive.'
        end
    end
    
    % interp must be one of:
    % 'bicubic', 'bilinear', 'nearest', or 'triangle'
    % but accept any shortening thereof.
    valid = {'bicubic', 'bilinear', 'nearest', 'triangle'};
    if isempty(params.interp)
        params.interp = 'bilinear';
    end
    ind = find(strncmpi(params.interp,valid,length(params.interp)));
    if (length(ind)==1)
        params.interp = valid{ind};
    else
        error(['Invalid interpolation method: ',params.interp])
    end
    
    % solver must be one of:
    %    'backslash', '\', 'symmlq', 'lsqr', or 'normal'
    % but accept any shortening thereof.
    valid = {'backslash', '\', 'symmlq', 'lsqr', 'normal'};
    if isempty(params.solver)
        params.solver = '\';
    end
    ind = find(strncmpi(params.solver,valid,length(params.solver)));
    if (length(ind)==1)
        params.solver = valid{ind};
    else
        error(['Invalid solver option: ',params.solver])
    end
    
    % extend must be one of:
    %    'never', 'warning', 'always'
    % but accept any shortening thereof.
    valid = {'never', 'warning', 'always'};
    if isempty(params.extend)
        params.extend = 'warning';
    end
    ind = find(strncmpi(params.extend,valid,length(params.extend)));
    if (length(ind)==1)
        params.extend = valid{ind};
    else
        error(['Invalid extend option: ',params.extend])
    end
    
    % tilesize == inf by default
    if isempty(params.tilesize)
        params.tilesize = inf;
    elseif (length(params.tilesize)>1) || (params.tilesize<3)
        error 'Tilesize must be scalar and > 0.'
    end
    
    % overlap == 0.20 by default
    if isempty(params.overlap)
        params.overlap = 0.20;
    elseif (length(params.overlap)>1) || (params.overlap<0) || (params.overlap>0.5)
        error 'Overlap must be scalar and 0 < overlap < 1.'
    end
    
    