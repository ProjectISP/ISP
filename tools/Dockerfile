FROM alpine:3.7
#FROM alpine:3.14
LABEL maintainer="Roberto Cabieces (popipopa@hotmail.com)"

RUN apk add --no-cache bash bash-doc bash-completion
RUN apk add --no-cache musl-dev 
RUN apk add --no-cache g++ gcc gdb make cmake
RUN apk add --no-cache gfortran
#RUN apk add --no-cache fort77
WORKDIR /source

# Build focmec
COPY ./focmec /source/focmec/
WORKDIR /source/focmec/src/
RUN bash build_package_sh
RUN cp /source/focmec/bin/* /bin
COPY ./focmec/test.inp /app/

# Build mti
COPY ./mti /source/mti/
WORKDIR /source/mti/
RUN bash compile_mti.sh
RUN cp bin/* /bin

#copy mti files to bin
RUN mkdir -p /app/mti
RUN cp -rf ./* /app/mti
RUN cp bin/* /app/mti
#RUN cp -rf ./* /app
COPY ./test /app
COPY ./test /app/mti

# Build nil
COPY ./nll /source/nll/
WORKDIR /source/nll/src/
#RUN cmake .
RUN mkdir bin
RUN make -R all
RUN cp bin/* /bin

# Remove source code
#RUN rm -rf /source

# Final workdir directory
WORKDIR /app
COPY ./test /bin
